from .local import LocalCDNStrategy
from .cloudinary import CloudinaryCDNStrategy
from .base import CDNStrategy
from functools import lru_cache

@lru_cache(maxsize=None)
def get_cdn_strategy(name: str) -> CDNStrategy:
    if name == "local":
        return LocalCDNStrategy()
    elif name == "CloudinaryCDNStrategy":
        return CloudinaryCDNStrategy()
    else:
        raise ValueError(f"Unknown CDN strategy: {name}")