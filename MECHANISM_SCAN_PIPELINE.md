# Mechanism Scan Pipeline

## Purpose

This file defines the current scanning pipeline for larger-model mechanism work.

It is intentionally split into two layers:

- an accurate starting point for `12B` work,
- a provisional legacy scan pipeline kept for data collection before the old
  `1B`-anchored scripts are refactored.

## Current Presets

### `mechanism_discovery_foundation`

This is the accurate starting point.

It currently runs:

- `exp_11_review_pack.py`

Use it when the goal is to stabilize labels and review artifacts before drawing
 any mechanism conclusion.

### `mechanism_scan_legacy`

This is the current scanning preset for collecting larger-model scan data while
 the old scan logic is still intact.

It runs:

- `exp_00_diagnosis.py`
- `exp_01_refusal.py`
- `exp_01b_cross_layer.py`
- `exp_16_safe_response_dictionary.py`
- `exp_17_gemma_scope_feature_probe.py`
- `exp_19_l17_l23_late_impact.py`

Alias:

- `scan_pipeline`

## Status

`mechanism_scan_legacy` is valid for collecting data, but it is not yet the
final accurate `12B` mechanism-discovery pipeline.

Reason:

- `exp_01_refusal.py` still assumes the old extraction/scan structure.
- `exp_01b_cross_layer.py` still organizes cross-layer tests around the old
  `L17/L23` story.
- `exp_16_safe_response_dictionary.py` still uses fixed `ABL_LAYERS = [17, 23]`.
- `exp_19_l17_l23_late_impact.py` still fixes `TARGET_LAYER = 17` and
  `ABL_LAYERS = [17, 23]`.

These scripts are kept unchanged for now so we can collect scan data first.

## Usage

Example:

```bash
python run_pipeline.py \
  --preset mechanism_scan_legacy \
  --run-name gemma3_12b_scan_main \
  --model google/gemma-3-12b-it \
  --hf-token "$HF_TOKEN" \
  --n-train 64 \
  --n-eval 2 \
  --scope-top-k-family 8 \
  --min-group-size 2
```

## Interpretation Rule

While this preset is active, read the outputs as:

- provisional scan evidence,
- not final `12B` layer conclusions.

The outputs should be used to decide how to refactor:

- `exp_18_l17_vector_quantification.py`
- `exp_19_l17_l23_late_impact.py`

after new `12B` discovery data is available.
