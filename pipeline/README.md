# Pipeline Package

This folder contains the stage and preset definitions for the `12B`-first
mechanism-rebuild workflow.

Layout:

- `catalog.py`: stage metadata, presets, and stage-summary rendering
- `__init__.py`: package exports for the top-level runner

Design split:

- `run_pipeline.py` is the execution wrapper
- `pipeline/` owns the workflow definition

Rules:

- `1B` experiments are references for workflow only
- `12B` stage ordering and blocked-stage rules live here
- `ShieldGemma` remains the default safety-judgement policy
