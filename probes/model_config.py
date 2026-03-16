from __future__ import annotations

from probes.model_adapter import get_config_adapter, get_model_adapter

def get_text_config(config):
    return get_config_adapter(config).text_config


def get_num_hidden_layers(config) -> int:
    return get_config_adapter(config).num_layers


def get_hidden_size(config) -> int:
    return get_config_adapter(config).hidden_size


def get_runtime_num_layers(model) -> int:
    return len(get_model_adapter(model).layers)


def ensure_model_layer_alignment(model) -> int:
    config_layers = get_num_hidden_layers(model.config)
    runtime_layers = get_runtime_num_layers(model)
    if config_layers != runtime_layers:
        raise ValueError(
            "Model layer mismatch: "
            f"config reports {config_layers} layers but runtime exposes {runtime_layers} layers. "
            "Refusing to continue because layer-index assumptions may be wrong."
        )
    return runtime_layers


def validate_layer_indices(
    model,
    layers: list[int] | tuple[int, ...],
    *,
    context: str = "",
) -> list[int]:
    total_layers = ensure_model_layer_alignment(model)
    normalized = [int(layer) for layer in layers]
    invalid = [layer for layer in normalized if layer < 0 or layer >= total_layers]
    if invalid:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}invalid layer indices {invalid}; "
            f"valid range is [0, {total_layers - 1}] for the loaded model."
        )
    return normalized
