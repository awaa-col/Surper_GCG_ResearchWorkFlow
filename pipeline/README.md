# Pipeline Package

This folder is the canonical home of the rebuilt `12B` research pipeline.

Rules:

- `1B` experiments are references for search order only.
- `12B` layer, vector, and closure claims must be rediscovered here.
- `ShieldGemma` is the default safety-judgement policy.
- Blocked stages may exist in the catalog, but only runnable stages may be used as the active mainline.

Current runnable chain:

- `T0` Eval Calibration
- `T1` Baseline Diagnosis
- `T2` Gate Discovery

Current blocked chain:

- `T3` through `T11`

Entry point:

- Use `run_pipeline.py`
