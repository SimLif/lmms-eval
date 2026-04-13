import time
from typing import List

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.batch_utils import run_with_oom_retry
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3_VLSimple
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_chat")
class Qwen3_VL(Qwen3_VLSimple):
    is_simple = False

    def _process_chunk(self, chunk):
        """Process one batch chunk through the model.

        Returns list of ``(clean_answer, context, gen_kwargs, n_tokens)``
        tuples — one per sample in *chunk*.
        """
        ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
        chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
        chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
        visuals = []
        videos = []
        for messages in chat_messages:
            visual, video, _ = messages.extract_media()
            visuals.append(visual)
            videos.append(video)
        visuals = self.flatten(visuals)
        videos = self.flatten(videos)
        gen_kwargs = all_gen_kwargs[0]

        video_kwargs = {
            "max_pixels": self.max_pixels,
            "min_pixels": self.min_pixels,
        }
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
            video_kwargs["max_frames"] = self.max_num_frames
        else:
            video_kwargs["nframes"] = self.max_num_frames
        batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
        texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs_qwen = process_vision_info(
            batched_messages,
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        video_kwargs = {**video_kwargs, **video_kwargs_qwen}
        for _k in ("nframes", "fps", "max_frames"):
            video_kwargs.pop(_k, None)

        video_metadatas = None
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = (
                list(video_inputs),
                list(video_metadatas),
            )

        if len(chunk) > 1:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                **video_kwargs,
                do_resize=False,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                **video_kwargs,
                do_resize=False,
                return_tensors="pt",
            )

        if self.device_map == "auto":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to(self.device)

        default_gen_kwargs = {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
        }
        current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
        pad_token_id = self.tokenizer.pad_token_id

        if current_gen_kwargs["temperature"] > 0:
            current_gen_kwargs["do_sample"] = True
        else:
            current_gen_kwargs["do_sample"] = False
            current_gen_kwargs["temperature"] = None
            current_gen_kwargs["top_p"] = None
            current_gen_kwargs["top_k"] = None

        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=current_gen_kwargs["do_sample"],
            temperature=current_gen_kwargs["temperature"],
            top_p=current_gen_kwargs["top_p"],
            num_beams=current_gen_kwargs["num_beams"],
            max_new_tokens=current_gen_kwargs["max_new_tokens"],
            top_k=current_gen_kwargs.get("top_k", None),
            use_cache=self.use_cache,
        )

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        results = []
        for ans, context, ids in zip(answers, texts, generated_ids_trimmed):
            clean_ans = parse_reasoning_model_answer(ans)
            results.append((clean_ans, context, gen_kwargs, len(ids)))
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        for chunk in chunks:
            start_time = time.time()
            results = run_with_oom_retry(self._process_chunk, chunk, self)
            e2e_latency += time.time() - start_time
            for clean_ans, context, gen_kwargs, n_toks in results:
                total_tokens += n_toks
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
            pbar.update(1)
        res = re_ords.get_original(res)

        avg_speed = total_tokens / e2e_latency if e2e_latency > 0 else 0
        metric_dict = {
            "total_tokens": total_tokens,
            "e2e_latency": e2e_latency,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        pbar.close()
        return res
