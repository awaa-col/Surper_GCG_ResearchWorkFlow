from __future__ import annotations

from probes.model_adapter import get_config_adapter

def get_text_config(config):
    return get_config_adapter(config).text_config


def get_num_hidden_layers(config) -> int:
    return get_config_adapter(config).num_layers


def get_hidden_size(config) -> int:
    return get_config_adapter(config).hidden_size
