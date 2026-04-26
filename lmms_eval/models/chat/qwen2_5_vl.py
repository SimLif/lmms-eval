import time
from typing import List

from loguru import logger as eval_logger
from tqdm import tqdm

try:
    import decord
except ImportError:
    decord = None

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.batch_utils import run_with_oom_retry
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl_chat")
class Qwen2_5_VL(Qwen2_5_VLSimple):
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
        else:
            if videos and decord is not None:
                try:
                    video_path = videos[0]
                    vr = decord.VideoReader(video_path)
                    video_total_frames = len(vr)
                    nframes = min(self.max_num_frames, video_total_frames)
                    nframes = (nframes // 2) * 2
                    nframes = max(2, nframes)
                    video_kwargs["nframes"] = nframes
                except Exception as e:
                    eval_logger.warning(f"Failed to probe video {videos[0]}: {e}, using default nframes")
                    video_kwargs["nframes"] = self.max_num_frames
            else:
                video_kwargs["nframes"] = self.max_num_frames
        batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
        texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(batched_messages)
        padding_side = "left" if len(chunk) > 1 else "right"
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side=padding_side,
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
        chunks = list(re_ords.get_batched(n=self.batch_size, batch_fn=None))
        num_iters = len(chunks)
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        e2e_latency = 0
        total_tokens = 0
        i = 0
        while i < len(chunks):
            cur_bs = self.batch_size_per_gpu
            if len(chunks[i]) > cur_bs:
                flat = [s for c in chunks[i:] for s in c]
                chunks[i:] = [flat[j:j + cur_bs] for j in range(0, len(flat), cur_bs)]
                pbar.total = len(chunks)
                pbar.refresh()
            start_time = time.time()
            results = run_with_oom_retry(self._process_chunk, chunks[i], self)
            e2e_latency += time.time() - start_time
            for clean_ans, context, gen_kwargs, n_toks in results:
                total_tokens += n_toks
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
            pbar.update(1)
            i += 1
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
