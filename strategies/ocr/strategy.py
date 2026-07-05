from app.strategies.ocr.paddleocr import PaddleOCRAPIStrategy
from app.strategies.ocr.base import OCRStrategy
from functools import lru_cache
from .MicrsostOcr import TrOCRStrategy

@lru_cache(maxsize=None)
def get_ocr_strategy(name: str , lang) -> OCRStrategy:
    if name == "TrOCRStrategy":
        return TrOCRStrategy(lang)
    else:
        raise ValueError(f"Unknown OCR strategy: {name}")