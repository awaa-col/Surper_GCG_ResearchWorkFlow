"""Pipeline package for the 12B-first mechanism-rebuild workflow."""

from .catalog import (
    ExperimentSpec,
    StageSpec,
    PIPELINE_PRESETS,
    PIPELINE_STAGES,
    flatten_stage_specs,
    print_preset_table,
    print_stage_table,
    render_stage_summary,
)

__all__ = [
    "ExperimentSpec",
    "StageSpec",
    "PIPELINE_PRESETS",
    "PIPELINE_STAGES",
    "flatten_stage_specs",
    "print_preset_table",
    "print_stage_table",
    "render_stage_summary",
]
