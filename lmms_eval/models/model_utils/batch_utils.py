"""OOM-safe batch processing with automatic chunk halving."""

import gc
from typing import Any, Callable, List, Union

import torch
from loguru import logger as eval_logger

# Default starting batch size for ``batch_size="auto"``
_AUTO_DEFAULT: int = 128


def parse_batch_size(
    batch_size: Union[int, str],
    default_auto: int = _AUTO_DEFAULT,
) -> int:
    """Parse a batch_size argument that may be int, ``"auto"``, or ``"auto:N"``.

    Returns a concrete integer batch size.
    """
    if isinstance(batch_size, int):
        return batch_size
    s = str(batch_size).strip()
    if s.startswith("auto"):
        parts = s.split(":")
        return int(parts[1]) if len(parts) > 1 else default_auto
    return int(s)


def run_with_oom_retry(
    process_fn: Callable[[List[Any]], List[Any]],
    chunk: List[Any],
    model: Any,
    min_batch_size: int = 1,
) -> List[Any]:
    """Run *process_fn* on *chunk* with automatic OOM recovery.

    On ``torch.cuda.OutOfMemoryError`` the failed sub-chunk is split in half
    and each half is retried independently.  ``model.batch_size_per_gpu`` is
    lowered so that **future** chunks produced by the caller also use the
    reduced size.

    Args:
        process_fn: ``fn(sub_chunk) -> list[result]`` — processes one
            (sub-)chunk and returns a flat list of per-sample results.
        chunk: The original chunk (list of sample tuples).
        model: Object with a writable ``batch_size_per_gpu`` attribute.
        min_batch_size: If OOM occurs at this size, re-raise immediately.

    Returns:
        Concatenated result lists from all successfully processed sub-chunks.
    """
    queue: list[list[Any]] = [list(chunk)]
    all_results: list[Any] = []

    while queue:
        sub = queue.pop(0)
        try:
            results = process_fn(sub)
            all_results.extend(results)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            n = len(sub)
            if n <= min_batch_size:
                eval_logger.error(
                    f"CUDA OOM with batch_size={n} (min={min_batch_size}), "
                    "cannot reduce further."
                )
                raise
            mid = n // 2
            new_bs = max(min_batch_size, mid)
            eval_logger.warning(
                f"CUDA OOM at batch_size={n}, halving to {new_bs}"
            )
            model.batch_size_per_gpu = new_bs
            # Re-insert halves at front so they are processed in order.
            queue.insert(0, sub[:mid])
            queue.insert(1, sub[mid:])

    return all_results
