# Colab Pipeline

This document describes the rebuilt `12B`-first mechanism-discovery workflow.

Core rule:

- `1B` is a workflow reference.
- `12B` must rebuild bottom theory from scratch.
- `ShieldGemma` stays the primary safety judge throughout.

## 1. Environment

In a fresh Colab runtime:

```bash
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content
!git clone https://github.com/awaa-col/Surper_GCG_ResearchWorkFlow.git Surper_GCG
%cd /content/Surper_GCG
!pip install -e .
```

If you need gated Hugging Face access:

```python
import os
os.environ["HF_TOKEN"] = "YOUR_TOKEN"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
```

Optional shared knobs:

```python
RUN_NAME = "gemma3_12b_main"
RESUME = "--resume"  # set to "" for a fresh run
```

## 2. Mainline Bootstrap

If you want the first clean `12B` mainline bootstrap, run:

```bash
!python run_pipeline.py \
  --preset t0_t1_bootstrap \
  --run-name "$RUN_NAME" \
  $RESUME
```

Today this preset runs:

- `exp_11_review_pack.py`
- `exp_00_diagnosis.py`

## 3. Gate Discovery

After `T0-T1` are complete, the next runnable chain is:

```bash
!python run_pipeline.py \
  --preset t0_t2_bootstrap \
  --run-name "$RUN_NAME" \
  $RESUME
```

This preset runs:

1. `T0 eval_calibration`
2. `T1 baseline_diagnosis`
3. `T2 gate_discovery`

If the run already exists after a Colab reset, restore:

```text
/content/Surper_GCG/results/pipeline_runs/gemma3_12b_main/
```

Make sure `exp11_review_pack.jsonl` and `exp00_diagnosis.json` are present, then rerun with `--resume`.

## 4. Support Tech

If you only want the review-pack support stage:

```bash
!python run_pipeline.py \
  --preset t0_eval_only \
  --run-name "${RUN_NAME}_eval_calibration" \
  $RESUME
```

## 5. Human Review Policy

The pipeline is intentionally hybrid.

After `T0`:

- review the exported review pack
- confirm ShieldGemma and manual semantics roughly align
- lock the stage-level audit rubric before moving on

After `T1`:

- inspect high-risk samples and malformed outputs
- confirm the model has a stable refusal object worth localizing
- write a short baseline memo before attempting gate discovery

Do not auto-advance into historical scan wrappers.

## 6. Stage Order

The intended `12B` stage order is:

1. `T0 eval_calibration`
2. `T1 baseline_diagnosis`
3. `T2 gate_discovery`
4. `T3 cross_layer_refinement`
5. `T4 detect_discovery`
6. `T5 late_safe_response_discovery`
7. `T6 family_feature_mapping`
8. `T7 candidate_quantification`
9. `T8 minimal_causal_closure`
10. `T9 robustness`
11. `T10 attack_acceptance`
12. `T11 trace_family_token_detail`

Only `T0-T2` are currently wired for direct execution.

## 7. Where Results Go

Outputs are written to:

```text
results/pipeline_runs/<timestamp>_<model_slug>_<preset>/
```

If you pass `--run-name some_fixed_name`, outputs go to:

```text
results/pipeline_runs/some_fixed_name/
```

Each run directory includes:

- experiment outputs
- `pipeline_manifest.json`
- `pipeline_stage_summary.md`

## 8. Practical Advice

- Prefer `t0_t1_bootstrap` as the first real mainline run.
- Use `t0_eval_only` only when you intentionally need a review pack.
- Use `t0_t2_bootstrap` when you want the current full runnable chain.
- If the run OOMs, lower `--n-train`, `--n-eval`, and `--max-new-tokens`.
- Keep the same `RUN_NAME` and add `--resume` after Colab restarts.

## 9. Transfer Caution

What may transfer from `1B`:

- there may be some upstream refusal gate
- there may be downstream safety-organization effects
- single-direction stories are probably incomplete

What must be rediscovered on `12B`:

- whether a dominant gate exists at all
- where that gate lives
- whether detect and exec still separate
- whether late safe-response structure exists and where
- whether stronger models produce cleaner unsafe execution instead of low-grade degeneration

Until those questions are answered, do not treat `L17`, `L23`, `r_exec`, or old family labels as valid `12B` theory.
