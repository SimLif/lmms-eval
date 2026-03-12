import os
from typing import Optional

from dotenv import load_dotenv
from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.llm_judge.base import ServerInterface

load_dotenv()

_server: Optional[ServerInterface] = None


def is_judge_enabled() -> bool:
    """Check whether the LLM judge is enabled via environment variable."""
    return os.getenv("USE_LLM_JUDGE", "false").lower() == "true"


def _get_judge_server() -> ServerInterface:
    """Return a singleton judge server instance."""
    global _server
    if _server is not None:
        return _server

    api_type = os.getenv("API_TYPE", "openai")
    model = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")

    config = ServerConfig(model_name=model, temperature=0, max_tokens=256)
    _server = get_server(server_name=api_type, config=config)
    return _server


def judge_binary(question: str, answer: str, prediction: str) -> float:
    """Use LLM judge to evaluate whether prediction matches answer.

    Returns:
        100.0 if the judge deems the prediction correct, 0.0 otherwise.
    """
    server = _get_judge_server()

    try:
        result = server.evaluate_binary(
            question=question,
            answer=answer,
            prediction=prediction,
            output_format="0/1",
        )
    except Exception as e:
        eval_logger.warning(f"LLM judge call failed: {e!s}")
        return 0.0

    # evaluate_binary returns {"result": int, "success": bool, ...}
    # With output_format="0/1": 1 = correct, 0 = incorrect
    if result.get("success") and result.get("result") is not None:
        return 100.0 if result["result"] == 1 else 0.0

    return 0.0
