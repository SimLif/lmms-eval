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


def get_judge_model_name() -> str:
    """Return the configured judge model name."""
    return os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")


def _get_judge_server() -> ServerInterface:
    """Return a singleton judge server instance."""
    global _server
    if _server is not None:
        return _server

    api_type = os.getenv("API_TYPE", "openai")
    model = get_judge_model_name()

    config = ServerConfig(model_name=model, temperature=0, max_tokens=256)
    _server = get_server(server_name=api_type, config=config)
    return _server


def judge_binary(question: str, answer: str, prediction: str) -> dict:
    """Use LLM judge to evaluate whether prediction matches answer.

    Returns:
        dict with keys:
            score: 100.0 if correct, 0.0 otherwise
            judge_model: model name used for judging
            judge_raw_response: raw response text from the judge
    """
    server = _get_judge_server()
    model_name = get_judge_model_name()

    try:
        result = server.evaluate_binary(
            question=question,
            answer=answer,
            prediction=prediction,
            output_format="0/1",
        )
    except Exception as e:
        eval_logger.warning(f"LLM judge call failed: {e!s}")
        return {"score": 0.0, "judge_model": model_name, "judge_raw_response": f"ERROR: {e!s}"}

    if result.get("success") and result.get("result") is not None:
        score = 100.0 if result["result"] == 1 else 0.0
    else:
        score = 0.0

    return {
        "score": score,
        "judge_model": result.get("model", model_name),
        "judge_raw_response": result.get("raw_response", ""),
    }
