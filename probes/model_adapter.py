from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelAdapter:
    model: object

    @classmethod
    def from_model(cls, model):
        return cls(model=model)

    @classmethod
    def from_config(cls, config):
        proxy = type("ConfigProxy", (), {})()
        proxy.config = config
        return cls(model=proxy)

    @property
    def config(self):
        return getattr(self.model, "config")

    @property
    def text_config(self):
        config = self.config
        for attr in ("text_config", "language_config", "llm_config"):
            nested = getattr(config, attr, None)
            if nested is not None:
                return nested
        return config

    @property
    def num_layers(self) -> int:
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(self.text_config, attr, None)
            if value is not None:
                return int(value)
        raise AttributeError(f"Could not infer num_hidden_layers from {type(self.config).__name__}")

    @property
    def hidden_size(self) -> int:
        for attr in ("hidden_size", "d_model", "n_embd"):
            value = getattr(self.text_config, attr, None)
            if value is not None:
                return int(value)
        raise AttributeError(f"Could not infer hidden_size from {type(self.config).__name__}")

    @property
    def text_model(self):
        candidates = [
            lambda m: getattr(getattr(m, "model", None), "language_model", None),
            lambda m: getattr(m, "language_model", None),
            lambda m: getattr(m, "model", None),
            lambda m: getattr(m, "text_model", None),
            lambda m: getattr(getattr(m, "model", None), "text_model", None),
        ]
        for candidate in candidates:
            text_model = candidate(self.model)
            if text_model is not None and hasattr(text_model, "layers"):
                return text_model
        raise AttributeError(f"Could not locate transformer text model for {type(self.model).__name__}")

    @property
    def layers(self):
        return self.text_model.layers

    def get_layer(self, layer_idx: int):
        return self.layers[layer_idx]

    @property
    def embed_tokens(self):
        text_model = self.text_model
        if hasattr(text_model, "embed_tokens"):
            return text_model.embed_tokens
        raise AttributeError(f"Could not locate embed_tokens for {type(self.model).__name__}")

    @property
    def has_vision_tower(self) -> bool:
        outer_model = getattr(self.model, "model", None)
        return hasattr(outer_model, "vision_tower") or hasattr(self.model, "vision_tower")

    @property
    def model_family(self) -> str:
        config_name = type(self.config).__name__.lower()
        model_name = type(self.model).__name__.lower()
        if "gemma3" in config_name or "gemma3" in model_name:
            return "gemma3"
        if "gemma" in config_name or "gemma" in model_name:
            return "gemma"
        return type(self.model).__name__


def get_model_adapter(model) -> ModelAdapter:
    return ModelAdapter.from_model(model)


def get_config_adapter(config) -> ModelAdapter:
    return ModelAdapter.from_config(config)
