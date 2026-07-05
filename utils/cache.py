import json
from redis import asyncio as aioredis
from app.config import settings

import redis

class ImageCache:
    _instance = None
    redis = None
    aioredis = None
    prefix = "cache:image"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def connectsync(cls):
        if cls.redis is None:
            cls.redis = redis.from_url(settings.redis_url, decode_responses=True)

    @classmethod
    def connectAsync(cls):
        if cls.aioredis is None:
            cls.aioredis = aioredis.from_url(settings.redis_url,   socket_timeout=20,   decode_responses=True)

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    # ── sync (Celery) ──────────────────────────────
    def get(self, key: str):
        data = self.redis.get(self._key(key))
        return json.loads(data) if data else None

    def set(self, key: str, result: dict, ttl: int = 86400):
        self.redis.setex(self._key(key), ttl, json.dumps(result))

    def delete(self, key: str):
        self.redis.delete(self._key(key))

    def acquire_lock(self, key: str, ttl: int = 3600) -> bool:
        return bool(self.redis.set(self._key(key), "1", nx=True, ex=ttl))


    # ── async (FastAPI) ────────────────────────────
    async def getAsync(self, key: str):
        data = await self.aioredis.get(self._key(key))
        return json.loads(data) if data else None

    async def setAsync(self, key: str, result: dict, ttl: int = 86400):
        await self.aioredis.setex(self._key(key), ttl, json.dumps(result))

    async def xread(self, streams: dict, block: int, count: int):
        return await self.aioredis.xread(streams, block=block, count=count)
   
    @classmethod
    async def close(cls):
        if cls.aioredis is not None:
            await cls.aioredis.close()  
        