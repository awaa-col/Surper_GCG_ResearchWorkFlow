from __future__ import annotations


def get_text_config(config):
    for attr in ("text_config", "language_config", "llm_config"):
        nested = getattr(config, attr, None)
        if nested is not None:
            return nested
    return config


def get_num_hidden_layers(config) -> int:
    text_config = get_text_config(config)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(text_config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"Could not infer num_hidden_layers from {type(config).__name__}")


def get_hidden_size(config) -> int:
    text_config = get_text_config(config)
    for attr in ("hidden_size", "d_model", "n_embd"):
        value = getattr(text_config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"Could not infer hidden_size from {type(config).__name__}")
