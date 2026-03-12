"""
Shared metrics for medical VQA tasks.

This module provides common evaluation metrics used across multiple
medical VQA tasks (VQA-RAD, SLAKE, PATH-VQA, etc.).

Text normalization constants are imported from vqa_eval_metric.py to avoid duplication.
"""

import re
from collections import defaultdict

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

# ==================== Text Normalization (shared with vqa_eval_metric.py) ====================

# Import constants from EvalAIAnswerProcessor for consistency
contractions = EvalAIAnswerProcessor.CONTRACTIONS
manual_map = EvalAIAnswerProcessor.NUMBER_MAP
articles = EvalAIAnswerProcessor.ARTICLES
period_strip = EvalAIAnswerProcessor.PERIOD_STRIP
comma_strip = EvalAIAnswerProcessor.COMMA_STRIP
punct = EvalAIAnswerProcessor.PUNCTUATIONS


def normalize_word(token: str) -> str:
    """Normalize text following VQA-v2 evaluation protocol.

    Args:
        token: Input text string

    Returns:
        Normalized text string
    """
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (re.search(comma_strip, token) is not None):
            _token = _token.replace(p, "")
        else:
            _token = _token.replace(p, " ")
    token = period_strip.sub("", _token, re.UNICODE)

    _token = []
    temp = token.lower().split()
    for word in temp:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            _token.append(word)
    for i, word in enumerate(_token):
        if word in contractions:
            _token[i] = contractions[word]
    token = " ".join(_token)
    token = token.replace(",", "")
    return token


# ==================== Helper Functions ====================


def split_sentence(sentence: str, n: int) -> dict[str, int]:
    """Split sentence into n-grams.

    Args:
        sentence: Input sentence
        n: N-gram size

    Returns:
        Dictionary mapping n-gram to count
    """
    words = defaultdict(int)
    tmp_sentence = sentence.lower().strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i : i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words


# ==================== Metrics ====================


def calculate_f1score(candidate: str, reference: str) -> tuple[float, float, float]:
    """Calculate token-level F1 score (recommended version).

    This version correctly handles word frequencies by counting overlaps
    as min(candidate_count, reference_count).

    Args:
        candidate: Predicted answer
        reference: Ground truth answer

    Returns:
        Tuple of (f1_score, precision, recall)
    """
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)

    word_set = set(candidate_words.keys()).union(reference_words.keys())

    tp = 0  # True Positive: overlapping tokens
    fp = 0  # False Positive: extra tokens in candidate
    fn = 0  # False Negative: missing tokens from reference

    for word in word_set:
        cand_count = candidate_words.get(word, 0)
        ref_count = reference_words.get(word, 0)
        common = min(cand_count, ref_count)  # Correct overlap count
        tp += common
        fp += cand_count - common
        fn += ref_count - common

    if sum(candidate_words.values()) == 0 or sum(reference_words.values()) == 0 or tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def calculate_f1score_old(candidate: str, reference: str) -> tuple[float, float, float]:
    """Calculate token-level F1 score (legacy version for backward compatibility).

    WARNING: This version has a bug where it only counts candidate word
    frequencies for overlapping words, leading to incorrect scores.

    This function is kept only for backward compatibility with existing
    evaluation results. New evaluations should use calculate_f1score().

    Args:
        candidate: Predicted answer
        reference: Ground truth answer

    Returns:
        Tuple of (f1_score, precision, recall)
    """
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)

    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]  # Bug: should use min(cand, ref)
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]

    if len(candidate_words) == 0:
        return 0.0, 0.0, 0.0
    elif len(reference_words) == 0:
        return 0.0, 0.0, 0.0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0.0, 0.0, 0.0
        else:
            return 2 * precision * recall / (precision + recall), precision, recall


def calculate_bleu(candidate: str, reference: str) -> float:
    """Calculate BLEU score using NLTK.

    Args:
        candidate: Predicted answer
        reference: Ground truth answer

    Returns:
        BLEU score (0.0 to 1.0)
    """
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_tokens = nltk.tokenize.word_tokenize(candidate)
    reference_tokens = nltk.tokenize.word_tokenize(reference)

    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
    return bleu_score


def calculate_exactmatch(candidate: str, reference: str) -> float:
    """Calculate exact match score (legacy, not recommended).

    WARNING: This metric has unclear semantics and is not a standard
    evaluation metric. It calculates: (# of reference word types in candidate) / (# of candidate tokens)

    This function is kept only for backward compatibility. Consider using
    F1 score or simple string equality instead.

    Args:
        candidate: Predicted answer
        reference: Ground truth answer

    Returns:
        Score between 0.0 and 1.0
    """
    candidate = normalize_word(candidate)
    reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]

    if total == 0:
        return 0.0
    else:
        return count / total
