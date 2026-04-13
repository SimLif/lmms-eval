"""Unified evaluation runner for lmms-eval medical benchmarks.

Reads model configs from a YAML file and runs evaluations via subprocess,
replacing the duplicated shell scripts (eval_med_mini_multi.sh, etc.).

Usage:
    uv run python scripts/run_eval.py -c scripts/configs/med_eval_mini.yaml
    uv run python scripts/run_eval.py -c scripts/configs/med_eval_mini.yaml --models Qwen3-VL-2B-Instruct
    uv run python scripts/run_eval.py -c scripts/configs/med_eval_mini.yaml --tags baseline --parallel
    uv run python scripts/run_eval.py -c scripts/configs/med_eval_mini.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Defaults:
    task: str = "med_eval_mini"
    num_gpus: int = 8
    accelerate_port: int = 12346
    batch_size: int = 32
    output_dir: str = "logs/med_eval_mini"
    vllm_tp: int = 8
    vllm_gpu_util: float = 0.80
    log_samples: bool = True
    skip_if_done: bool = False
    default_transformers_version: str = "4.57.6"
    default_vllm_version: str = "0.11.0"
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str
    model_type: str
    pretrained: str
    launch: str  # "multi" | "single" | "vllm"
    transformers_version: str | None = None
    batch_size: int | None = None
    max_pixels: int | None = None
    min_pixels: int | None = None
    attn_implementation: str | None = None
    enable_thinking: bool = False
    max_thinking_tokens: int | None = None
    gen_kwargs: dict[str, Any] | None = None
    vllm_tp: int | None = None
    vllm_gpu_util: float | None = None
    gpu: int | None = None
    tags: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class ResolvedModel:
    """ModelConfig merged with Defaults — all fields are concrete."""

    name: str
    model_type: str
    pretrained: str
    launch: str
    task: str
    num_gpus: int
    accelerate_port: int
    batch_size: int
    output_dir: str
    vllm_tp: int
    vllm_gpu_util: float
    log_samples: bool
    skip_if_done: bool
    default_transformers_version: str
    default_vllm_version: str
    transformers_version: str | None
    max_pixels: int | None
    min_pixels: int | None
    attn_implementation: str | None
    enable_thinking: bool
    max_thinking_tokens: int | None
    gen_kwargs: dict[str, Any] | None
    gpu: int | None
    tags: list[str]
    env: dict[str, str]


@dataclass
class Result:
    name: str
    success: bool
    skipped: bool = False
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> tuple[Defaults, list[ModelConfig]]:
    with open(path) as f:
        raw = yaml.safe_load(f)

    raw_defaults = raw.get("defaults", {})
    defaults = Defaults(
        task=raw_defaults.get("task", "med_eval_mini"),
        num_gpus=raw_defaults.get("num_gpus", 8),
        accelerate_port=raw_defaults.get("accelerate_port", 12346),
        batch_size=raw_defaults.get("batch_size", 32),
        output_dir=raw_defaults.get("output_dir", "logs/med_eval_mini"),
        vllm_tp=raw_defaults.get("vllm_tp", 8),
        vllm_gpu_util=raw_defaults.get("vllm_gpu_util", 0.80),
        log_samples=raw_defaults.get("log_samples", True),
        skip_if_done=raw_defaults.get("skip_if_done", False),
        default_transformers_version=raw_defaults.get(
            "default_transformers_version", "4.57.6"
        ),
        default_vllm_version=raw_defaults.get("default_vllm_version", "0.11.0"),
        env=raw_defaults.get("env", {}),
    )

    models: list[ModelConfig] = []
    for m in raw.get("models", []):
        models.append(
            ModelConfig(
                name=m["name"],
                model_type=m["model_type"],
                pretrained=m["pretrained"],
                launch=m["launch"],
                transformers_version=m.get("transformers_version"),
                batch_size=m.get("batch_size"),
                max_pixels=m.get("max_pixels"),
                min_pixels=m.get("min_pixels"),
                attn_implementation=m.get("attn_implementation"),
                enable_thinking=m.get("enable_thinking", False),
                max_thinking_tokens=m.get("max_thinking_tokens"),
                gen_kwargs=m.get("gen_kwargs"),
                vllm_tp=m.get("vllm_tp"),
                vllm_gpu_util=m.get("vllm_gpu_util"),
                gpu=m.get("gpu"),
                tags=m.get("tags", []),
                env=m.get("env", {}),
            )
        )

    return defaults, models


# ---------------------------------------------------------------------------
# Model selection & resolution
# ---------------------------------------------------------------------------


def select_models(
    models: list[ModelConfig],
    names: list[str] | None,
    tags: list[str] | None,
    exclude: list[str] | None,
) -> list[ModelConfig]:
    selected = models

    if names:
        name_set = set(names)
        selected = [m for m in selected if m.name in name_set]

    if tags:
        tag_set = set(tags)
        selected = [m for m in selected if tag_set & set(m.tags)]

    if exclude:
        exclude_set = set(exclude)
        selected = [m for m in selected if m.name not in exclude_set]

    return selected


def resolve_model(model: ModelConfig, defaults: Defaults) -> ResolvedModel:
    merged_env = {**defaults.env, **model.env}
    return ResolvedModel(
        name=model.name,
        model_type=model.model_type,
        pretrained=model.pretrained,
        launch=model.launch,
        task=defaults.task,
        num_gpus=defaults.num_gpus,
        accelerate_port=defaults.accelerate_port,
        batch_size=model.batch_size if model.batch_size is not None else defaults.batch_size,
        output_dir=defaults.output_dir,
        vllm_tp=model.vllm_tp if model.vllm_tp is not None else defaults.vllm_tp,
        vllm_gpu_util=(
            model.vllm_gpu_util
            if model.vllm_gpu_util is not None
            else defaults.vllm_gpu_util
        ),
        log_samples=defaults.log_samples,
        skip_if_done=defaults.skip_if_done,
        default_transformers_version=defaults.default_transformers_version,
        default_vllm_version=defaults.default_vllm_version,
        transformers_version=model.transformers_version,
        max_pixels=model.max_pixels,
        min_pixels=model.min_pixels,
        attn_implementation=model.attn_implementation,
        enable_thinking=model.enable_thinking,
        max_thinking_tokens=model.max_thinking_tokens,
        gen_kwargs=model.gen_kwargs,
        gpu=model.gpu,
        tags=model.tags,
        env=merged_env,
    )


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def setup_environment() -> None:
    venv_cudnn = Path(".venv/lib/python3.10/site-packages/nvidia/cudnn/lib")
    if venv_cudnn.exists():
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{venv_cudnn}:{ld}"

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    if not os.environ.get("HF_TOKEN"):
        log.warning("HF_TOKEN is not set — private datasets may fail to load")


# ---------------------------------------------------------------------------
# Transformers version management
# ---------------------------------------------------------------------------

_current_tf_version: str | None = None

# Companion package versions for each transformers version.
# transformers 4.49.0 requires tokenizers<0.22, so we must downgrade it.
_TRANSFORMERS_COMPANIONS: dict[str, dict[str, str]] = {
    "4.49.0": {"tokenizers": "0.21.1"},
}


def get_current_transformers_version() -> str:
    result = subprocess.run(
        [".venv/bin/python", "-c", "import transformers; print(transformers.__version__)"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def ensure_transformers_version(
    want: str | None,
    default_tf: str,
    default_vllm: str,
) -> None:
    """Switch transformers version via `uv pip install --no-deps`.

    Does NOT modify pyproject.toml or uv.lock. Does NOT trigger uv sync,
    so flash-attn and other CUDA extensions are preserved.

    For versions with known transitive dependency gaps (e.g., tokenizers),
    swaps those packages too. When restoring to default, restores companions.
    """
    global _current_tf_version

    if want is None:
        return

    if _current_tf_version is None:
        _current_tf_version = get_current_transformers_version()

    if _current_tf_version == want:
        return

    log.info("Switching transformers %s → %s", _current_tf_version, want)

    # Build package list: transformers + any companions for this version
    packages = [f"transformers=={want}"]
    companions = _TRANSFORMERS_COMPANIONS.get(want, {})
    for pkg, ver in companions.items():
        packages.append(f"{pkg}=={ver}")

    # If restoring to default and the previous version had companions, restore them
    if want == default_tf and _current_tf_version in _TRANSFORMERS_COMPANIONS:
        # Query current lockfile versions to restore
        restore = _get_lockfile_versions(
            list(_TRANSFORMERS_COMPANIONS[_current_tf_version].keys())
        )
        for pkg, ver in restore.items():
            packages.append(f"{pkg}=={ver}")

    cmd = ["uv", "pip", "install", "--no-deps"] + packages
    subprocess.run(cmd, check=True, capture_output=True)
    _current_tf_version = want


def _get_lockfile_versions(package_names: list[str]) -> dict[str, str]:
    """Read pinned versions from uv.lock for the given packages."""
    versions: dict[str, str] = {}
    try:
        with open("uv.lock") as f:
            content = f.read()
        for name in package_names:
            # Parse: name = "tokenizers"\nversion = "0.22.2"
            import re

            pattern = rf'^name = "{re.escape(name)}"\nversion = "([^"]+)"'
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                versions[name] = match.group(1)
    except FileNotFoundError:
        log.warning("uv.lock not found, cannot determine restore versions")
    return versions


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def build_model_args(model: ResolvedModel) -> str:
    if model.launch == "vllm":
        parts = [
            f"model={model.pretrained}",
            f"tensor_parallel_size={model.vllm_tp}",
            f"gpu_memory_utilization={model.vllm_gpu_util}",
            "disable_log_stats=True",
        ]
        if model.max_pixels is not None:
            parts.append(
                f'mm_processor_kwargs={{"max_pixels":{model.max_pixels}}}'
            )
        return ",".join(parts)

    # multi / single
    parts = [f"pretrained={model.pretrained}"]
    if model.max_pixels is not None:
        parts.append(f"max_pixels={model.max_pixels}")
    if model.min_pixels is not None:
        parts.append(f"min_pixels={model.min_pixels}")
    if model.attn_implementation is not None:
        parts.append(f"attn_implementation={model.attn_implementation}")
    if model.enable_thinking:
        parts.append("enable_thinking=True")
        if model.max_thinking_tokens is not None:
            parts.append(f"max_thinking_tokens={model.max_thinking_tokens}")
    return ",".join(parts)


def build_gen_kwargs(model: ResolvedModel) -> str | None:
    if not model.gen_kwargs:
        return None
    return ",".join(f"{k}={v}" for k, v in model.gen_kwargs.items())


def build_command(model: ResolvedModel, limit: int | None = None) -> list[str]:
    model_output_dir = f"{model.output_dir}/{model.name}"
    model_args = build_model_args(model)
    gen_kwargs = build_gen_kwargs(model)

    # Determine the --model argument
    cli_model = "vllm" if model.launch == "vllm" else model.model_type

    # Common lmms_eval arguments
    lmms_args = [
        "-m", "lmms_eval",
        "--model", cli_model,
        "--model_args", model_args,
        "--tasks", model.task,
        "--batch_size", str(model.batch_size),
        "--output_path", model_output_dir,
    ]
    if model.log_samples:
        lmms_args.append("--log_samples")
    if gen_kwargs:
        lmms_args.extend(["--gen_kwargs", gen_kwargs])
    if limit is not None:
        lmms_args.extend(["--limit", str(limit)])

    # Build the full command based on launch mode
    if model.launch == "multi":
        cmd = [
            ".venv/bin/accelerate", "launch",
            f"--num_processes={model.num_gpus}",
            f"--main_process_port={model.accelerate_port}",
            *lmms_args,
        ]
    elif model.launch in ("single", "vllm"):
        cmd = [".venv/bin/python", *lmms_args]
    else:
        raise ValueError(f"Unknown launch mode: {model.launch}")

    return cmd


def build_subprocess_env(model: ResolvedModel) -> dict[str, str]:
    env = {**os.environ, **model.env}
    if model.launch == "single" and model.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(model.gpu)
    return env


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------


def check_results_exist(output_dir: str, model_name: str) -> bool:
    model_dir = Path(output_dir) / model_name
    return any(model_dir.glob("*_results.json")) if model_dir.exists() else False


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_single_model(
    model: ResolvedModel,
    limit: int | None,
    dry_run: bool,
) -> Result:
    model_output_dir = Path(model.output_dir) / model.name

    # Skip if done
    if model.skip_if_done and check_results_exist(model.output_dir, model.name):
        log.info("[SKIP] %s — results already exist", model.name)
        return Result(name=model.name, success=True, skipped=True)

    # Version switch
    if not dry_run:
        ensure_transformers_version(
            model.transformers_version,
            model.default_transformers_version,
            model.default_vllm_version,
        )

    cmd = build_command(model, limit)
    env = build_subprocess_env(model)

    if dry_run:
        env_prefix = ""
        if model.launch == "single" and model.gpu is not None:
            env_prefix = f"CUDA_VISIBLE_DEVICES={model.gpu} "
        extra_env = {
            k: v for k, v in model.env.items() if k not in os.environ
        }
        if extra_env:
            env_prefix += " ".join(f"{k}={v}" for k, v in extra_env.items()) + " "
        log.info("[DRY-RUN] %s\n  %s%s", model.name, env_prefix, " ".join(cmd))
        return Result(name=model.name, success=True)

    # Ensure output dir
    model_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_output_dir / f"{model.name}.log"

    log.info(
        "━━━ [%s] launch=%s batch=%d ━━━",
        model.name, model.launch, model.batch_size,
    )

    t0 = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    duration = time.time() - t0

    if proc.returncode == 0:
        log.info("[PASS] %s (%.0fs)", model.name, duration)
    else:
        log.error("[FAIL] %s (exit=%d, %.0fs) — see %s", model.name, proc.returncode, duration, log_path)

    return Result(
        name=model.name,
        success=proc.returncode == 0,
        duration=duration,
    )


def run_sequential(
    models: list[ResolvedModel],
    limit: int | None,
    dry_run: bool,
) -> list[Result]:
    results = []
    for model in models:
        result = run_single_model(model, limit, dry_run)
        results.append(result)
    return results


def run_parallel(
    models: list[ResolvedModel],
    limit: int | None,
    dry_run: bool,
) -> list[Result]:
    # Validate: all models must have gpu field
    for m in models:
        if m.gpu is None:
            raise ValueError(
                f"Parallel mode requires 'gpu' field for every model, "
                f"but '{m.name}' has no gpu set"
            )

    # Validate: all models must use the same transformers version
    versions = {m.transformers_version for m in models}
    if len(versions) > 1:
        raise ValueError(
            f"Parallel mode requires all models to use the same "
            f"transformers version, got: {versions}"
        )

    # Version switch once
    first = models[0]
    if not dry_run:
        ensure_transformers_version(
            first.transformers_version,
            first.default_transformers_version,
            first.default_vllm_version,
        )

    if dry_run:
        return run_sequential(models, limit, dry_run)

    # Launch all processes
    procs: list[tuple[str, subprocess.Popen, Path]] = []
    for model in models:
        model_output_dir = Path(model.output_dir) / model.name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        log_path = model_output_dir / f"{model.name}.log"

        cmd = build_command(model, limit)
        env = build_subprocess_env(model)

        log.info("[PARALLEL] Launching %s on GPU %s", model.name, model.gpu)
        log_f = open(log_path, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((model.name, proc, log_path))

    # Wait and collect
    results = []
    for name, proc, log_path in procs:
        t0 = time.time()
        proc.wait()
        duration = time.time() - t0
        success = proc.returncode == 0
        if success:
            log.info("[PASS] %s (%.0fs)", name, duration)
        else:
            log.error("[FAIL] %s (exit=%d) — see %s", name, proc.returncode, log_path)
        results.append(Result(name=name, success=success, duration=duration))

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: list[Result]) -> None:
    passed = [r for r in results if r.success and not r.skipped]
    failed = [r for r in results if not r.success]
    skipped = [r for r in results if r.skipped]

    log.info("━━━ Summary ━━━")
    if passed:
        log.info("PASSED (%d):", len(passed))
        for r in passed:
            log.info("  ✓ %s (%.0fs)", r.name, r.duration)
    if skipped:
        log.info("SKIPPED (%d):", len(skipped))
        for r in skipped:
            log.info("  ○ %s", r.name)
    if failed:
        log.info("FAILED (%d):", len(failed))
        for r in failed:
            log.info("  ✗ %s (%.0fs)", r.name, r.duration)
    log.info(
        "Total: %d passed, %d failed, %d skipped",
        len(passed), len(failed), len(skipped),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified evaluation runner for lmms-eval medical benchmarks.",
    )
    p.add_argument(
        "-c", "--config", required=True,
        help="Path to YAML config file",
    )
    p.add_argument(
        "--models", type=lambda s: s.split(","),
        help="Run only these models (comma-separated names)",
    )
    p.add_argument(
        "--tags", type=lambda s: s.split(","),
        help="Run only models with these tags (comma-separated)",
    )
    p.add_argument(
        "--exclude", type=lambda s: s.split(","),
        help="Exclude these models (comma-separated names)",
    )
    p.add_argument(
        "--parallel", action="store_true",
        help="Run models in parallel (each must have 'gpu' field)",
    )
    p.add_argument(
        "--skip-if-done", action="store_true",
        help="Skip models with existing *_results.json",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Pass --limit N to lmms_eval for quick testing",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    p.add_argument(
        "--task",
        help="Override the task from config",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch_size for all models",
    )
    p.add_argument(
        "--output-dir",
        help="Override output directory",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    defaults, models = load_config(args.config)

    # CLI overrides
    if args.task:
        defaults.task = args.task
    if args.output_dir:
        defaults.output_dir = args.output_dir
    if args.skip_if_done:
        defaults.skip_if_done = True

    # Select models
    selected = select_models(models, args.models, args.tags, args.exclude)
    if not selected:
        log.error("No models matched the selection criteria")
        sys.exit(1)

    log.info(
        "Config: %s | Task: %s | Models: %d selected",
        args.config, defaults.task, len(selected),
    )

    # Resolve models (merge defaults + per-model overrides + CLI overrides)
    resolved = []
    for m in selected:
        r = resolve_model(m, defaults)
        if args.batch_size is not None:
            r.batch_size = args.batch_size
        resolved.append(r)

    # Setup environment
    setup_environment()

    # Run with crash-safe version restore
    try:
        if args.parallel:
            results = run_parallel(resolved, args.limit, args.dry_run)
        else:
            results = run_sequential(resolved, args.limit, args.dry_run)
    finally:
        # Always restore default transformers version on exit
        if not args.dry_run and _current_tf_version != defaults.default_transformers_version:
            log.info("Restoring transformers to default %s", defaults.default_transformers_version)
            ensure_transformers_version(
                defaults.default_transformers_version,
                defaults.default_transformers_version,
                defaults.default_vllm_version,
            )

    # Summary
    print_summary(results)

    # Exit with failure if any model failed
    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
