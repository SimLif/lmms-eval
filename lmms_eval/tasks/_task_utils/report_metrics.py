"""
Shared metrics for medical report generation tasks (MIMIC-CXR, IU-XRAY).

Metrics:
    - BLEU-4: sentence-level with smoothing (Chen & Cherry, 2014)
    - ROUGE-L: via Google's rouge-score library
    - METEOR: via NLTK (considers synonyms and stemming)
    - RaTEScore: medical-domain NER + semantic similarity (optional,
      per-sample via ``calculate_single_ratescore``)

Tokenization uses nltk.word_tokenize for proper punctuation handling.
"""

import logging
from typing import Any

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


def _ensure_nltk_resources() -> None:
    """Download NLTK data required for METEOR and tokenization if missing."""
    resources = {
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_resources()

# Singleton ROUGE scorer (thread-safe, reusable)
_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

# BLEU smoothing for sentence-level evaluation
_smoothing = SmoothingFunction().method1


def calculate_bleu_4(candidate: str, reference: str) -> float:
    """Calculate sentence-level BLEU-4 with smoothing.

    Uses nltk.word_tokenize for proper punctuation separation and
    SmoothingFunction.method1 to avoid zero scores when any n-gram
    order has no matches (standard for sentence-level BLEU).

    Args:
        candidate: Model-generated report text.
        reference: Ground-truth report text.

    Returns:
        BLEU-4 score in [0.0, 1.0].
    """
    candidate_tokens = word_tokenize(candidate.lower())
    reference_tokens = word_tokenize(reference.lower())

    if not candidate_tokens or not reference_tokens:
        return 0.0

    try:
        return sentence_bleu(
            [reference_tokens],
            candidate_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=_smoothing,
        )
    except Exception:
        return 0.0


def calculate_rouge_l(candidate: str, reference: str) -> float:
    """Calculate ROUGE-L F1 score via Google's rouge-score library.

    Args:
        candidate: Model-generated report text.
        reference: Ground-truth report text.

    Returns:
        ROUGE-L F1 score in [0.0, 1.0].
    """
    if not candidate.strip() or not reference.strip():
        return 0.0

    scores = _rouge_scorer.score(reference, candidate)
    return scores["rougeL"].fmeasure


def calculate_meteor(candidate: str, reference: str) -> float:
    """Calculate METEOR score via NLTK.

    METEOR considers synonyms (via WordNet) and stemming, making it
    more suitable for medical text where paraphrasing is common
    (e.g., "cardiomegaly" vs "enlarged heart").

    Args:
        candidate: Model-generated report text.
        reference: Ground-truth report text.

    Returns:
        METEOR score in [0.0, 1.0].
    """
    candidate_tokens = word_tokenize(candidate.lower())
    reference_tokens = word_tokenize(reference.lower())

    if not candidate_tokens or not reference_tokens:
        return 0.0

    try:
        return single_meteor_score(reference_tokens, candidate_tokens)
    except Exception:
        return 0.0


def compute_report_metrics(candidate: str, reference: str) -> dict[str, float]:
    """Compute all report generation metrics for a single sample.

    All scores are scaled to [0, 100] for consistency with other
    medical evaluation tasks (VQA accuracy, F1, etc.).

    Args:
        candidate: Model-generated report text.
        reference: Ground-truth report text.

    Returns:
        Dict with keys: bleu_4, rouge_l, meteor (each in [0, 100]).
    """
    return {
        "bleu_4": calculate_bleu_4(candidate, reference) * 100,
        "rouge_l": calculate_rouge_l(candidate, reference) * 100,
        "meteor": calculate_meteor(candidate, reference) * 100,
    }


# ==================== RaTEScore (optional) ====================

_ratescore_model: Any = None
_ratescore_available: bool | None = None


def _get_ratescore() -> Any:
    """Lazy-load RaTEScore model. Returns None if unavailable."""
    global _ratescore_model, _ratescore_available

    if _ratescore_available is False:
        return None
    if _ratescore_model is not None:
        return _ratescore_model

    try:
        from RaTEScore import RaTEScore

        _ratescore_model = RaTEScore(
            bert_model="Angelakeke/RaTE-NER-Deberta",
            eval_model="FremyCompany/BioLORD-2023-C",
        )
        _ratescore_available = True
        logger.info("RaTEScore loaded successfully.")
        return _ratescore_model
    except ImportError as e:
        _ratescore_available = False
        logger.warning(
            "RaTEScore import failed: %s. "
            "Install with: uv add RaTEScore medspacy. "
            "Skipping RaTEScore metric.",
            e,
        )
        return None
    except Exception as e:
        _ratescore_available = False
        logger.warning("RaTEScore failed to load: %s. Skipping.", e)
        return None


def calculate_single_ratescore(candidate: str, reference: str) -> float:
    """Compute RaTEScore for a single (candidate, reference) pair.

    The RaTEScore model is lazy-loaded on first call and reused as a
    singleton.  Returns 0.0 when the library is not installed or
    computation fails.

    Score is scaled to [0, 100] for consistency with other metrics.

    Args:
        candidate: Model-generated report text.
        reference: Ground-truth report text.

    Returns:
        RaTEScore in [0.0, 100.0], or 0.0 if unavailable.
    """
    scorer = _get_ratescore()
    if scorer is None:
        return 0.0

    try:
        scores = scorer.compute_score([candidate], [reference])
        return float(scores[0]) * 100
    except Exception as e:
        logger.warning("RaTEScore computation failed: %s", e)
        return 0.0


def calculate_ratescore(candidates: list[str], references: list[str]) -> list[float]:
    """Compute RaTEScore for a batch of samples.

    RaTEScore is a medical report-aware metric that uses NER to extract
    medical entities and computes semantic similarity. It works best in
    batch mode.

    Scores are scaled to [0, 100] for consistency with other metrics.

    Args:
        candidates: List of model-generated reports.
        references: List of ground-truth reports.

    Returns:
        List of per-sample scores in [0, 100], or empty list if unavailable.
    """
    scorer = _get_ratescore()
    if scorer is None:
        return []

    try:
        scores = scorer.compute_score(candidates, references)
        return [float(s) * 100 for s in scores]
    except Exception as e:
        logger.warning("RaTEScore computation failed: %s", e)
        return []
