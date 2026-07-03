# app/Worker/model_registry.py

_models = {}
_strategy_registry = {}


def register_strategy(cls):
    """Decorator: add to app/strategies/base.py or each strategy file.
    Registers a strategy class under its own class name so it can be
    resolved dynamically from a string (e.g. from .env)."""
    _strategy_registry[cls.__name__] = cls
    return cls

def get_strategy_class(name: str):
    try:
        return _strategy_registry[name]
    except KeyError:
        raise ValueError(
            f"Unknown strategy '{name}'. Registered strategies: "
            f"{list(_strategy_registry.keys())}"
        )

def set_model(key, value):
    _models[key] = value

def get_model(key):
    return _models[key]