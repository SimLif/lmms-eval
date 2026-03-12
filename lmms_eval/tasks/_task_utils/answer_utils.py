"""
Answer extraction utilities for reasoning models.

This module provides functions to extract answers from model outputs
in various formats (boxed, answer tags, etc.) without external dependencies.
"""

from __future__ import annotations

import re


def extract_boxed_answer(predict_str: str) -> str:
    """Extract the answer from \\boxed{} format, handling nested braces.

    Args:
        predict_str: The prediction string containing the boxed answer.

    Returns:
        The extracted answer from \\boxed{}, or an empty string if not found.
    """
    boxed_start = "\\boxed{"
    start_indices = []

    # Find all positions where \boxed{ starts
    pos = 0
    while True:
        pos = predict_str.find(boxed_start, pos)
        if pos == -1:
            break
        start_indices.append(pos)
        pos += 1

    if not start_indices:
        return ""

    # For each \boxed{ occurrence, find the matching closing brace
    results = []
    for start_pos in start_indices:
        brace_count = 0
        pos = start_pos + len(boxed_start) - 1  # Position at the opening brace

        while pos < len(predict_str):
            char = predict_str[pos]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    content_start = start_pos + len(boxed_start)
                    content = predict_str[content_start:pos]
                    results.append(content)
                    break
            pos += 1

    # Return the last (rightmost) match if multiple found
    return results[-1] if results else ""


def extract_answer_tag(predict_str: str) -> str:
    """Extract the answer from <answer> tags or \\boxed{} format.

    Args:
        predict_str: The prediction string containing the answer.

    Returns:
        The extracted answer, or an empty string if not found.
    """
    # First try to extract from <answer> tags
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match_result = re.search(pattern, predict_str)
    if match_result:
        return match_result.group(1)

    # If no <answer> tag found, try to extract from \boxed{} format
    boxed_answer = extract_boxed_answer(predict_str)
    if boxed_answer:
        return boxed_answer

    return ""


def parse_reasoning_answer(model_answer: str, strict: bool = False) -> str:
    """Parse answer from reasoning model output.

    This is a drop-in replacement for reasoning_model_utils.parse_reasoning_model_answer
    with improved \boxed{} handling for nested braces.

    Args:
        model_answer: Raw model output
        strict: If True, return empty string when no tags found (for think mode)
                If False, return original answer (for non-think mode)

    Returns:
        Extracted answer or original text
    """
    extracted = extract_answer_tag(model_answer)
    if extracted:
        return extracted.strip()
    return "" if strict else model_answer


def parse_multi_choice_response(
    response: str,
    all_choices: list[str] | None = None,
) -> str:
    """Parse multiple choice answer from model response.

    Extract the predicted option letter (e.g., A, B, C, D) from model output.
    This function only matches letter patterns, not answer text content.

    Matching priority:
    1. (A) (B) (C) (D) - parenthesized format
    2. " A " " B " - space-surrounded format
    3. "A." "B." - dot-suffixed format

    Args:
        response: Model output text (should be pre-processed with parse_reasoning_answer)
        all_choices: List of valid choice letters, defaults to ["A", "B", "C", "D"]

    Returns:
        Matched choice letter (uppercase), or empty string "" if no match found.
        When multiple candidates found, returns the last occurring one.
    """
    if all_choices is None:
        all_choices = ["A", "B", "C", "D"]

    # Normalize: strip punctuation and add spaces
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    candidates = []
    ans_with_brack = False

    # Priority 1: Match (A) (B) (C) (D) format
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    # Priority 2: Match " A " " B " format (space-surrounded)
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Priority 3: Match "A." "B." format
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    # Return result
    if len(candidates) == 0:
        return ""  # No match found
    elif len(candidates) > 1:
        # Multiple candidates: return the last occurring one
        start_indexes = []
        if ans_with_brack:
            for can in candidates:
                index = response.rfind(f"({can})")
                start_indexes.append(index)
        else:
            for can in candidates:
                index = response.rfind(f" {can} ")
                if index == -1:
                    index = response.rfind(f"{can}.")
                start_indexes.append(index)
        return candidates[start_indexes.index(max(start_indexes))]
    else:
        return candidates[0]
