"""FGMoE-Qwen3VL model for lmms-eval.

Inherits all evaluation logic from Qwen3_VL and only overrides the model
loading to use FGMoEQwen3VLForConditionalGeneration via trust_remote_code.

Usage in med_eval_mini.yaml:
  - name: FGMoE-v13-step300
    model_type: fgmoe_qwen3_vl
    pretrained: /path/to/slim_checkpoint
    launch: multi
    params: "~2B"
    max_pixels: 1003520
    min_pixels: 200704
    attn_implementation: sdpa
    tags: [fgmoe]
"""

from typing import Optional, Union

import torch
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL


@register_model("fgmoe_qwen3_vl")
class FGMoEQwen3VL(Qwen3_VL):
    """FGMoE-Qwen3VL: loads via trust_remote_code from a slim checkpoint.

    The slim checkpoint must have been created with:
        uv run python scripts/slim_checkpoint.py <ckpt> --experiment <exp> --base-model <base>

    It contains:
      - model.safetensors (FGMoE weights)
      - config.json with fgmoe_config + auto_map
      - modeling_fgmoe_qwen3vl.py (trust_remote_code entry point)
      - modeling/ (bundled source files)
      - tokenizer files (from base model)
    """

    def __init__(
        self,
        pretrained: str,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 200704,
        max_pixels: int = 1003520,
        **kwargs,
    ) -> None:
        # Call grandparent (lmms base) to skip Qwen3_VL's __init__
        # We only want to reuse generate/evaluate methods, not the model loading
        from lmms_eval.api.model import lmms as lmms_base
        lmms_base.__init__(self)

        from accelerate import Accelerator, DistributedType
        from loguru import logger as eval_logger
        from lmms_eval.models.model_utils.batch_utils import parse_batch_size

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        eval_logger.info(f"Loading FGMoE checkpoint: {pretrained}")

        # Load class directly via importlib to bypass auto_map format issues
        # (AFS checkpoints may have auto_map with >1 dot which breaks
        #  transformers 4.57.6's get_class_from_dynamic_module)
        import importlib.util
        import sys
        _entry = f"{pretrained}/modeling_fgmoe_qwen3vl.py"
        _spec = importlib.util.spec_from_file_location("modeling_fgmoe_qwen3vl", _entry)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["modeling_fgmoe_qwen3vl"] = _mod
        _spec.loader.exec_module(_mod)
        _ModelCls = _mod.FGMoEQwen3VLForConditionalGeneration

        self._model = _ModelCls.from_pretrained(pretrained, **model_kwargs).eval()
        # Note: FGMoELayer.forward already returns a plain tensor when not self.training
        # (set by .eval() above), so no unwrap patch is needed here.

        self.pretrained = pretrained
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = 32

        self.processor = AutoProcessor.from_pretrained(
            pretrained, max_pixels=max_pixels, min_pixels=min_pixels
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = "You are a helpful assistant."
        self.interleave_visuals = False
        self.reasoning_prompt = None
        self.use_custom_video_loader = False
        self.fps = None
        self.max_image_size = None

        self._config = self._model.config
        self._max_length = 2048
        self.batch_size_per_gpu = parse_batch_size(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
