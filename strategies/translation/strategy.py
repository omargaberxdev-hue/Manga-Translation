from .base import TranslationStrategy
from functools import lru_cache
from .Hunyuan import HunyuanTranslation
from .OpenRouter import OpenRouterTranslation
@lru_cache(maxsize=None)
def get_translation_stratgy(name: str) -> TranslationStrategy:
    if name =="hunyusn":
        return HunyuanTranslation()
    if name =="OpenRouterTranslation":
        return OpenRouterTranslation()
    else:
        raise ValueError(f"Unknown CDN strategy: {name}")                    