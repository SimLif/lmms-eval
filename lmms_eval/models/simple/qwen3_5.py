import re
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers.generation import LogitsProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)


class ThinkingTokenBudgetProcessor(LogitsProcessor):
    """Force the model to emit ``</think>`` after *max_thinking_tokens*
    tokens have been generated inside a ``<think>…</think>`` block.

    This lets the model keep its chain-of-thought ability while capping the
    computational cost of the reasoning phase.  Supports batched generation.
    """

    def __init__(
        self,
        start_think_token_id: int,
        end_think_token_id: int,
        max_thinking_tokens: int,
    ) -> None:
        self.start_think_token_id = start_think_token_id
        self.end_think_token_id = end_think_token_id
        self.max_thinking_tokens = max_thinking_tokens

    def reset(self, batch_size: int) -> None:
        """Reset state before each ``model.generate()`` call.

        Assumes the model is already inside a thinking block (the chat
        template emits ``<think>`` as part of the prompt, so the very
        first generated token is already "thinking").
        """
        self._thinking_count = [0] * batch_size
        self._in_thinking = [True] * batch_size

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        # Lazy init if reset() was not called
        if not hasattr(self, "_in_thinking") or len(self._in_thinking) != batch_size:
            self.reset(batch_size)

        for b in range(batch_size):
            last_token = input_ids[b, -1].item()
            if last_token == self.start_think_token_id:
                self._in_thinking[b] = True
                self._thinking_count[b] = 0
            elif last_token == self.end_think_token_id:
                self._in_thinking[b] = False

            if self._in_thinking[b]:
                self._thinking_count[b] += 1
                if self._thinking_count[b] >= self.max_thinking_tokens:
                    scores[b, :] = float("-inf")
                    scores[b, self.end_think_token_id] = 0.0
        return scores

# Qwen3.5 requires transformers >= 5.x
# Import will fail on older versions; the model won't be registered
try:
    from transformers import (
        AutoProcessor,
        AutoTokenizer,
        Qwen3_5ForConditionalGeneration,
    )

    _has_qwen3_5 = True
except ImportError:
    _has_qwen3_5 = False
    eval_logger.warning(
        "Failed to import Qwen3_5ForConditionalGeneration. "
        "Please upgrade transformers to >= 5.x: pip install transformers>=5.0.0"
    )

# Try to import MoE variant (for models like Qwen3.5-35B-A3B)
try:
    from transformers import Qwen3_5MoeForConditionalGeneration

    _has_qwen3_5_moe = True
except ImportError:
    _has_qwen3_5_moe = False

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_5")
class Qwen3_5(lmms):
    """
    Qwen3.5 Model - Native Multimodal Vision-Language Model
    https://huggingface.co/Qwen/Qwen3.5-4B

    Requires transformers >= 5.x
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        enable_thinking: bool = False,
        max_thinking_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if not _has_qwen3_5:
            raise ImportError(
                "Qwen3.5 requires transformers >= 5.x. "
                "Please upgrade: pip install transformers>=5.0.0"
            )

        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # Check whether its an MoE model (e.g., Qwen3.5-35B-A3B)
        match = re.search(r"A\d+B", pretrained)
        if match and _has_qwen3_5_moe:
            model_fn = Qwen3_5MoeForConditionalGeneration
        else:
            model_fn = Qwen3_5ForConditionalGeneration

        self._model = model_fn.from_pretrained(pretrained, **model_kwargs).eval()
        self.pretrained = pretrained
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self.enable_thinking = enable_thinking

        # Build a logits processor that caps the number of thinking tokens
        self._thinking_processor: Optional[ThinkingTokenBudgetProcessor] = None
        if enable_thinking and max_thinking_tokens is not None:
            start_id = self._tokenizer.convert_tokens_to_ids("<think>")
            end_id = self._tokenizer.convert_tokens_to_ids("</think>")
            self._thinking_processor = ThinkingTokenBudgetProcessor(
                start_think_token_id=start_id,
                end_think_token_id=end_id,
                max_thinking_tokens=int(max_thinking_tokens),
            )
            eval_logger.info(
                f"Thinking budget: max {max_thinking_tokens} tokens "
                f"(<think>={start_id}, </think>={end_id})"
            )

        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    def _resolve_model_name_for_cache(self) -> str:
        name = self.pretrained
        if self.enable_thinking:
            name += "-thinking"
        return name

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3.5")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, tasks, splits = zip(*chunk)
            # Use per-item task/split for visual lookup because the Collator
            # groups by gen_kwargs, so a single batch may mix different tasks.
            visual_list = [
                doc_to_visual[i](self.task_dict[tasks[i]][splits[i]][ids])
                for i, ids in enumerate(doc_id)
            ]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper to prevent truncation
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                            vr = decord.VideoReader(visual)
                            processed_visuals.append(
                                {
                                    "type": "video",
                                    "video": visual,
                                    "max_pixels": self.max_pixels,
                                    "min_pixels": self.min_pixels,
                                }
                            )
                        elif isinstance(visual, Image.Image):
                            processed_visuals.append(
                                {
                                    "type": "image",
                                    "image": visual,
                                    "max_pixels": self.max_pixels,
                                    "min_pixels": self.min_pixels,
                                }
                            )

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)
            texts = self.processor.apply_chat_template(
                batched_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            image_inputs, video_inputs = process_vision_info(
                batched_messages,
                return_video_kwargs=False,
                image_patch_size=16,
                return_video_metadata=False,
            )
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                indices = np.unique(indices)
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)
                video_inputs[0] = video_inputs[0][indices]
            if self.batch_size > 1:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
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
                    do_resize=False,
                    return_tensors="pt",
                )
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Set default generation kwargs
            # Qwen3.5 recommended: temperature=1.0, top_p=0.95, presence_penalty=1.5 for thinking mode
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

            # Build logits_processors list for thinking budget
            logits_processor_list = []
            if self._thinking_processor is not None:
                self._thinking_processor.reset(len(texts))
                logits_processor_list.append(self._thinking_processor)

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
                logits_processor=logits_processor_list or None,
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=not self.enable_thinking,
                clean_up_tokenization_spaces=False,
            )
            for i, ans in enumerate(answers):
                # In thinking mode, extract only the answer after </think>
                if self.enable_thinking and "</think>" in ans:
                    ans = ans.split("</think>", 1)[1].strip()
                elif self.enable_thinking and "<think>" in ans:
                    # Thinking was truncated without </think> — no answer
                    ans = ""
                # Clean residual special tokens from skip_special_tokens=False
                if self.enable_thinking:
                    ans = re.sub(r"<\|[^|]*\|>", "", ans).strip()
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for i, (ans, context, did) in enumerate(zip(answers, contexts, doc_id)):
                clean_ans = parse_reasoning_model_answer(ans)
                res.append(clean_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                self.cache_response(did, tasks[i], clean_ans)
                pbar.update(1)

        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
