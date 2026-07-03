import torch
from celery.signals import worker_init, worker_process_init

import app.strategies  # import triggers @register_strategy on every strategy class

from app.utils.cache import ImageCache
from .model_registry import set_model, get_strategy_class
from app.config import settings


def _load_strategy_model(strategy_name: str, use_gpu: bool):
    strategy_cls = get_strategy_class(strategy_name)

    if not hasattr(strategy_cls, "load_model"):
        return None

    model = strategy_cls.load_model()

    if not use_gpu and hasattr(strategy_cls, "share_memory"):
        strategy_cls.share_memory(model)

    return model


def _load_models(use_gpu: bool):
    device = "cuda" if use_gpu else "cpu"

    detection_model = _load_strategy_model(settings.detection_strategy, use_gpu)
    ocr_model = _load_strategy_model(settings.ocr_strategy, use_gpu)
    inpaint_model = _load_strategy_model(settings.Inpainting, use_gpu)
    _load_strategy_model(settings.translation_strategy, use_gpu)  

    set_model("inpaint", inpaint_model)
    set_model("detection", detection_model)
    set_model("extraction", ocr_model)
    set_model("device", device)

    print(f"Models loaded on {device}.")


@worker_init.connect
def load_models_main_process(**kwargs):
    if not torch.cuda.is_available():
        _load_models(use_gpu=False)
        print("Models loaded on CPU. Consider using GPU for better performance.")


@worker_process_init.connect
def load_models_per_worker(**kwargs):
    """Runs in each forked child process — required for GPU models."""
    if torch.cuda.is_available():
        _load_models(use_gpu=True)
    ImageCache.connectsync()