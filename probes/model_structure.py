from __future__ import annotations

from probes.model_adapter import get_model_adapter

def get_text_model(model):
    return get_model_adapter(model).text_model


def get_transformer_layers(model):
    return get_model_adapter(model).layers


def get_transformer_layer(model, layer_idx: int):
    return get_model_adapter(model).get_layer(layer_idx)


def get_embed_tokens_module(model):
    return get_model_adapter(model).embed_tokens
