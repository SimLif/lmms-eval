import transformers

from .language_model.llava_llama import LlavaLlamaConfig, LlavaLlamaForCausalLM
from .language_model.llava_llama_moe import (
    MoELLaVALlamaConfig,
    MoELLaVALlamaForCausalLM,
)
from .language_model.llava_qwen import LlavaQWenConfig, LlavaQWenForCausalLM
from .language_model.llava_qwen_moe import MoELLaVAQWenConfig, MoELLaVAQWenForCausalLM

a, b, c = transformers.__version__.split(".")[:3]
if a == "4" and int(b) >= 34:
    from .language_model.llava_mistral import (
        LlavaMistralConfig,
        LlavaMistralForCausalLM,
    )
    from .language_model.llava_mistral_moe import (
        MoELLaVAMistralConfig,
        MoELLaVAMistralForCausalLM,
    )
if a == "4" and int(b) >= 36:
    from .language_model.llava_minicpm import (
        LlavaMiniCPMConfig,
        LlavaMiniCPMForCausalLM,
    )
    from .language_model.llava_minicpm_moe import (
        MoELLaVAMiniCPMConfig,
        MoELLaVAMiniCPMForCausalLM,
    )
    from .language_model.llava_phi import LlavaPhiConfig, LlavaPhiForCausalLM
    from .language_model.llava_phi_moe import MoELLaVAPhiConfig, MoELLaVAPhiForCausalLM
    from .language_model.llava_stablelm import (
        LlavaStablelmConfig,
        LlavaStablelmForCausalLM,
    )
    from .language_model.llava_stablelm_moe import (
        MoELLaVAStablelmConfig,
        MoELLaVAStablelmForCausalLM,
    )
if a == "4" and int(b) >= 37:
    from .language_model.llava_qwen1_5 import (
        LlavaQwen1_5Config,
        LlavaQwen1_5ForCausalLM,
    )
    from .language_model.llava_qwen1_5_moe import (
        MoELLaVAQwen1_5Config,
        MoELLaVAQwen1_5ForCausalLM,
    )
if a == "4" and int(b) <= 31:
    from .language_model.llava_mpt import LlavaMPTConfig, LlavaMPTForCausalLM
