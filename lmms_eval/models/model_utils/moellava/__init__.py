import transformers

from .model import LlavaLlamaForCausalLM, LlavaQWenForCausalLM, MoELLaVALlamaForCausalLM

a, b, c = transformers.__version__.split(".")[:3]
if a == "4" and int(b) >= 34:
    from .model import LlavaMistralForCausalLM, MoELLaVAMistralForCausalLM
if a == "4" and int(b) >= 36:
    from .model import (
        LlavaMiniCPMForCausalLM,
        LlavaPhiForCausalLM,
        LlavaStablelmForCausalLM,
        MoELLaVAMiniCPMForCausalLM,
        MoELLaVAPhiForCausalLM,
        MoELLaVAStablelmForCausalLM,
    )
if a == "4" and int(b) >= 37:
    from .model import LlavaQwen1_5ForCausalLM, MoELLaVAQwen1_5ForCausalLM
