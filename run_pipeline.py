from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline import (
    ExperimentSpec,
    PIPELINE_PRESETS,
    PIPELINE_STAGES,
    flatten_stage_specs,
    print_preset_table,
    print_stage_table,
    render_stage_summary,
)


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "pipeline_runs"


def script_text(script_relpath: str) -> str:
    return (ROOT / script_relpath).read_text(encoding="utf-8")


def has_flag(script_relpath: str, flag: str) -> bool:
    text = script_text(script_relpath)
    escaped = re.escape(flag)
    patterns = [
        rf"add_argument\(\s*\"{escaped}\"",
        rf"add_argument\(\s*'{escaped}'",
    ]
    return any(re.search(pattern, text, flags=re.MULTILINE) for pattern in patterns)


def parse_repeated_kv(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE format, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in extra arg: {item}")
        parsed[key] = value
    return parsed


def build_model_slug(model_name: str) -> str:
    slug = model_name.replace("/", "__").replace(":", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in slug)


def build_run_name(preset: str, model_name: str, run_name: str | None) -> str:
    if run_name:
        return run_name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{build_model_slug(model_name)}_{preset}"


def build_command(
    spec: ExperimentSpec,
    *,
    python_bin: str,
    model: str,
    hf_token: str | None,
    output_path: Path,
    seed: int | None,
    n_train: int | None,
    n_eval: int | None,
    max_new_tokens: int | None,
    extra_args: dict[str, str],
) -> list[str]:
    cmd = [python_bin, spec.script]
    if has_flag(spec.script, "--model"):
        cmd.extend(["--model", model])
    if has_flag(spec.script, "--shield_audit"):
        cmd.append("--shield_audit")
    if has_flag(spec.script, "--shield_device"):
        cmd.extend(["--shield_device", os.getenv("SUPER_GCG_SHIELD_DEVICE", "auto")])
    if has_flag(spec.script, "--shield_truncate"):
        cmd.extend(["--shield_truncate", os.getenv("SUPER_GCG_SHIELD_TRUNCATE", "500")])
    if has_flag(spec.script, "--output") and not spec.use_default_output:
        cmd.extend(["--output", str(output_path)])
    if has_flag(spec.script, "--output_csv") and not spec.use_default_output:
        cmd.extend(["--output_csv", str(output_path.with_suffix(".csv"))])
    if has_flag(spec.script, "--output_jsonl") and not spec.use_default_output:
        cmd.extend(["--output_jsonl", str(output_path.with_suffix(".jsonl"))])
    if has_flag(spec.script, "--output_blind") and not spec.use_default_output:
        blind_path = output_path.with_name(output_path.stem + "_blind.json")
        cmd.extend(["--output_blind", str(blind_path)])
    if hf_token and has_flag(spec.script, "--hf_token"):
        cmd.extend(["--hf_token", hf_token])

    alias_map = spec.arg_aliases or {}
    common_values = {
        "seed": seed,
        "n_train": n_train,
        "n_eval": n_eval,
        "max_new_tokens": max_new_tokens,
    }
    for key, value in common_values.items():
        alias = alias_map.get(key, "")
        if alias and value is not None:
            cmd.extend([alias, str(value)])
    for key, value in extra_args.items():
        cmd.extend([key, value])
    cmd.extend(spec.default_args)
    return cmd


def validate_stage_selection(stage_keys: tuple[str, ...]) -> None:
    seen: set[str] = set()
    for stage_key in stage_keys:
        stage = PIPELINE_STAGES[stage_key]
        missing = [dep for dep in stage.depends_on if dep not in seen]
        if missing:
            raise ValueError(
                f"Stage {stage.tech_id} ({stage.key}) is selected before its prerequisites: {', '.join(missing)}"
            )
        seen.add(stage_key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the rebuilt 12B research pipeline.")
    parser.add_argument("--preset", default="t0_t1_bootstrap", choices=sorted(PIPELINE_PRESETS))
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append raw KEY=VALUE pairs. Use this only for script-specific options.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.list_presets:
        print_preset_table()
        return 0
    if args.list_stages:
        print_stage_table()
        return 0

    selected_stage_keys = PIPELINE_PRESETS[args.preset]
    validate_stage_selection(selected_stage_keys)
    preset_specs = flatten_stage_specs(selected_stage_keys)
    extra_args = parse_repeated_kv(args.extra_arg)

    run_name = build_run_name(args.preset, args.model, args.run_name)
    results_root = Path(args.results_root).resolve()
    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "pipeline_family": "12b_rebuild",
        "preset": args.preset,
        "selected_stage_keys": list(selected_stage_keys),
        "model": args.model,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "cwd": str(ROOT),
        "python_bin": args.python_bin,
        "hf_token_provided": bool(args.hf_token),
        "extra_args": extra_args,
        "shieldgemma_policy": "ShieldGemma-first; heuristic labels are non-authoritative.",
        "stages": [],
        "experiments": [],
    }

    for stage_key in selected_stage_keys:
        stage = PIPELINE_STAGES[stage_key]
        manifest["stages"].append(
            {
                "key": stage.key,
                "tech_id": stage.tech_id,
                "order": stage.order,
                "title": stage.title,
                "objective": stage.objective,
                "why_now": stage.why_now,
                "automation_mode": stage.automation_mode,
                "shieldgemma_policy": stage.shieldgemma_policy,
                "human_review_required": stage.human_review_required,
                "human_review_points": list(stage.human_review_points),
                "experiment_ids": list(stage.experiment_ids),
                "depends_on": list(stage.depends_on),
                "runnable_now": stage.runnable_now,
                "blocked_reason": stage.blocked_reason,
            }
        )

    for idx, spec in enumerate(preset_specs, start=1):
        output_path = run_dir / spec.output_name
        cmd = build_command(
            spec,
            python_bin=args.python_bin,
            model=args.model,
            hf_token=args.hf_token,
            output_path=output_path,
            seed=args.seed,
            n_train=args.n_train,
            n_eval=args.n_eval,
            max_new_tokens=args.max_new_tokens,
            extra_args=extra_args,
        )
        manifest["experiments"].append(
            {
                "index": idx,
                "stage": spec.stage,
                "script": spec.script,
                "output": str(output_path),
                "use_default_output": spec.use_default_output,
                "command": cmd,
            }
        )

    manifest_path = run_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stage_summary_path = run_dir / "pipeline_stage_summary.md"
    stage_summary_path.write_text(render_stage_summary(selected_stage_keys), encoding="utf-8")

    print(f"[pipeline] preset={args.preset}")
    print(f"[pipeline] model={args.model}")
    print(f"[pipeline] run_dir={run_dir}")
    print(f"[pipeline] manifest={manifest_path}")
    print(f"[pipeline] stage_summary={stage_summary_path}")
    print("[pipeline] policy=ShieldGemma-first")

    for item in manifest["experiments"]:
        command = item["command"]
        pretty = " ".join(shlex.quote(part) for part in command)
        print(f"\n[pipeline] stage={item['stage']} script={item['script']}")
        print(f"[pipeline] cmd: {pretty}")
        if args.dry_run:
            continue
        output_path = Path(str(item["output"]))
        if args.resume and output_path.exists():
            print(f"[pipeline] resume: skip existing output {output_path.name}")
            continue
        env = os.environ.copy()
        env.setdefault("SUPER_GCG_ENABLE_SHIELD_AUDIT", "1")
        env.setdefault("SUPER_GCG_SHIELD_DEVICE", "auto")
        env.setdefault("SUPER_GCG_SHIELD_TRUNCATE", "500")
        env.setdefault("SUPER_GCG_SHIELD_SUCCESS_KEY", "direct_danger_fewshot")
        env.setdefault("SUPER_GCG_SHIELD_SUCCESS_THRESHOLD", "0.5")
        env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        env.setdefault("TRANSFORMERS_VERBOSITY", "error")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        completed = subprocess.run(command, cwd=ROOT, check=False, env=env)
        if completed.returncode != 0:
            print(f"[pipeline] failed: {item['script']} exit={completed.returncode}")
            return completed.returncode

    if args.dry_run:
        print("\n[pipeline] dry-run complete")
    else:
        print("\n[pipeline] completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
