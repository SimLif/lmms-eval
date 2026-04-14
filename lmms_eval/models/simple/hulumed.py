from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.batch_utils import (
    parse_batch_size,
    run_with_oom_retry,
)


@register_model("hulumed")
class HuluMed(lmms):
    """Hulu-Med VL models (ZJU-AI4H/Hulu-Med-*).

    LLaVA-style architecture with custom vision encoder, loaded via
    trust_remote_code=True. Supports both Qwen2 and Qwen3 backbone variants.
    """

    def __init__(
        self,
        pretrained: str = "ZJU-AI4H/Hulu-Med-4B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
        ).eval()
        self.pretrained = pretrained

        self.processor = AutoProcessor.from_pretrained(
            pretrained, trust_remote_code=True
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True
        )
        self.system_prompt = system_prompt

        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = parse_batch_size(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

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
        raise NotImplementedError("Loglikelihood is not implemented for HuluMed")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _process_chunk(self, chunk):
        """Process one batch chunk through the model.

        Processes each sample individually through the custom HuluMed
        processor and model (batch_size=1 per inference call).
        """
        contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
        task = task[0]
        split = split[0]
        visual_list = [
            doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
        ]
        gen_kwargs = all_gen_kwargs[0]

        until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
        if isinstance(until, str):
            until = [until]
        elif not isinstance(until, list):
            raise ValueError(
                f"Expected `gen_kwargs['until']` to be of type "
                f"Union[str, list], but got {type(until)}"
            )
        until = [item for item in until if item != "\n\n"]

        if isinstance(contexts, tuple):
            contexts = list(contexts)
        for i in range(len(contexts)):
            if "<image>" in contexts[i]:
                contexts[i] = contexts[i].replace("<image>", "")

        default_gen_kwargs = {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": None,
            "num_beams": 1,
        }
        current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}

        if current_gen_kwargs["temperature"] > 0:
            current_gen_kwargs["do_sample"] = True
        else:
            current_gen_kwargs["do_sample"] = False
            current_gen_kwargs["temperature"] = None
            current_gen_kwargs["top_p"] = None

        results = []
        for i, context in enumerate(contexts):
            # Build conversation messages
            messages = []
            if self.system_prompt:
                messages.append(
                    {"role": "system", "content": self.system_prompt}
                )

            content = []
            if visual_list[i] is not None:
                for visual in visual_list[i]:
                    if isinstance(visual, Image.Image):
                        content.append({"type": "image", "image": visual})
            content.append({"type": "text", "text": context})
            messages.append({"role": "user", "content": content})

            # Process through HulumedProcessor
            inputs = self.processor(
                conversation=messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            # Cast pixel_values to model dtype (processor returns float32,
            # model is bfloat16)
            if inputs.get("pixel_values") is not None:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    dtype=torch.bfloat16
                )

            # Generate — pass multimodal tensors as explicit kwargs
            output_ids = self.model.generate(
                pixel_values=inputs.get("pixel_values"),
                grid_sizes=inputs.get("grid_sizes"),
                merge_sizes=inputs.get("merge_sizes"),
                modals=inputs.get("modals"),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            # Decode — model.generate() with inputs_embeds returns only
            # newly generated tokens (no input prefix), so decode directly.
            answer = self.processor.decode(
                output_ids[0], skip_special_tokens=True
            )

            for term in until:
                if len(term) > 0:
                    answer = answer.split(term)[0]

            results.append((answer, contexts[i], doc_id[i], gen_kwargs, task))
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            results = run_with_oom_retry(self._process_chunk, chunk, self)
            for answer, context, did, gen_kwargs, task in results:
                res.append(answer)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), answer
                )
                self.cache_response(did, task, answer)
                pbar.update(1)
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
