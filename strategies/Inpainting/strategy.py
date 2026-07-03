from app.strategies.ocr.paddleocr import PaddleOCRAPIStrategy
from app.strategies.ocr.base import OCRStrategy
from functools import lru_cache
from .LaMe import LaMeInpainting

@lru_cache(maxsize=None)
def get_ocr_strategy(name: str) -> OCRStrategy:
    if name == "LaMeInpainting":
        return LaMeInpainting()
    else:
        raise ValueError(f"Unknown Inpainting strategy: {name}")