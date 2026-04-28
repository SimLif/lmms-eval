import re
from typing import Any, Dict, Optional, Tuple, Union

from .prompt import (
    BINARY_JUDGE_PROMPT,
    COMPARATIVE_JUDGE_PROMPT,
    CORRECTNESS_JUDGE_PROMPT,
)


class JudgePromptBuilder:
    """Helper class to build prompts for different judge types"""

    @staticmethod
    def build_binary_prompt(question: str, answer: str, prediction: str, output_format: str = "0/1", custom_prompt: Optional[str] = None, **kwargs) -> str:
        """Build prompt for binary evaluation"""
        if custom_prompt:
            return custom_prompt.format(question=question, answer=answer, pred=prediction, prediction=prediction, **kwargs)

        if output_format == "0/1":
            positive, negative = ("0", "1")    # MedEvalKit: 0=correct
        elif output_format == "1/0":
            positive, negative = ("1", "0")    # legacy strict: 1=correct
        else:
            positive, negative = ("Yes", "No")

        return BINARY_JUDGE_PROMPT.format(question=question, answer=answer, prediction=prediction, positive=positive, negative=negative)

    @staticmethod
    def build_comparative_prompt(
        question: str, response1: str, response2: str, context: Optional[str] = None, score_range: Tuple[int, int] = (1, 10), custom_prompt: Optional[str] = None, evaluation_instruction: Optional[str] = None, **kwargs
    ) -> str:
        """Build prompt for comparative evaluation"""
        if custom_prompt:
            return custom_prompt.format(question=question, response1=response1, response2=response2, context=context or "", **kwargs)

        context_section = f"[Context]\n{context}\n\n" if context else ""

        if not evaluation_instruction:
            evaluation_instruction = f"Please provide scores from {score_range[0]} to {score_range[1]}."

        return COMPARATIVE_JUDGE_PROMPT.format(question=question, response1=response1, response2=response2, context_section=context_section, min_score=score_range[0], max_score=score_range[1], evaluation_instruction=evaluation_instruction)

    @staticmethod
    def build_correctness_prompt(question: str, answer: str, prediction: str, output_format: str = "yes/no", **kwargs) -> str:
        """Build prompt for correctness evaluation"""
        positive, negative = ("Yes", "No") if output_format == "yes/no" else ("1", "0")

        return CORRECTNESS_JUDGE_PROMPT.format(question=question, answer=answer, prediction=prediction, positive=positive, negative=negative)


class ResponseParser:
    """Helper class to parse different types of judge responses"""

    @staticmethod
    def parse_binary_response(response: str, output_format: str = "0/1") -> Union[int, bool]:
        """Parse binary response (0/1 or yes/no).

        MedEvalKit convention: 0 = correct, 1 = incorrect.
        Internal return: 1 = correct, 0 = incorrect (unchanged for downstream).

        Prefer an explicit ``<judge>X</judge>`` tag when present;
        otherwise fall back to pattern matching over the full response.
        """
        raw = response or ""
        tag_match = re.search(r"<judge>\s*([^<\s]+?)\s*</judge>", raw, re.IGNORECASE)
        tagged = tag_match.group(1).strip().lower() if tag_match else None

        response = raw.strip().lower()

        if output_format == "0/1" or output_format == "1/0":
            # Determine which raw value means "correct"
            correct_val = "0" if output_format == "0/1" else "1"
            incorrect_val = "1" if output_format == "0/1" else "0"

            if tagged is not None:
                if tagged in (correct_val, "correct", "true", "yes"):
                    return 1
                if tagged in (incorrect_val, "incorrect", "false", "no"):
                    return 0
            if any(pattern in response for pattern in [f"[{correct_val}]", f"score: {correct_val}", f"answer: {correct_val}"]):
                return 1
            if re.search(rf"<judge>\s*{correct_val}\s*</judge>", response):
                return 1
            # Default to incorrect when ambiguous
            return 0
        else:
            # yes/no format
            if tagged is not None:
                return tagged.startswith("yes") or tagged == "0"
            return response == "yes" or response.startswith("yes")

    @staticmethod
    def parse_score_response(response: str, score_range: Optional[Tuple[float, float]] = None) -> float:
        """Parse a single score from response"""
        try:
            # Try to extract first number from response
            numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
            if numbers:
                score = float(numbers[0])
                # Clamp to valid range if provided
                if score_range:
                    score = max(score_range[0], min(score, score_range[1]))
                return score
        except Exception:
            pass

        # Return minimum score as default
        return score_range[0] if score_range else 0.0

    @staticmethod
    def parse_comparative_response(response: str) -> Tuple[float, float]:
        """Parse comparative scores from response"""
        try:
            # Extract scores from first line
            lines = response.strip().split("\n")
            if lines:
                score_line = lines[0]
                # Handle different separators
                score_line = score_line.replace(",", " ").replace(";", " ")
                scores = re.findall(r"-?\d+(?:\.\d+)?", score_line)

                if len(scores) >= 2:
                    return float(scores[0]), float(scores[1])
        except Exception:
            pass

        return -1.0, -1.0

    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        """Parse JSON response"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                import json

                return json.loads(json_match.group())
        except Exception:
            pass

        return {}
