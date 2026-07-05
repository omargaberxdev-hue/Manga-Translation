import aiofiles
import os
from abc import ABC, abstractmethod

from app.config import settings

import cloudinary
from cloudinary import CloudinaryImage
import cloudinary.uploader
import cloudinary.api

# Import to format the JSON responses
# ==============================
import json


import aiofiles
import os
import asyncio
from abc import ABC, abstractmethod
from io import BytesIO

import cloudinary
import cloudinary.uploader

from .base import CDNStrategy

class CloudinaryCDNStrategy(CDNStrategy):
    
    def __init__(self):
        cloudinary.config(
            cloud_name=settings.cloudinary_cloud_name,
            api_key= settings.cloudinary_api_key,
            api_secret= settings.cloudinary_api_secret,
            secure=True
        )

    async def upload(self, data: bytes, filename: str) -> str:
        loop = asyncio.get_event_loop()

        def _upload():
            return cloudinary.uploader.upload(
                BytesIO(data),              #  correct way to pass bytes
                public_id=filename,         # file name (without extension issues)
                resource_type="auto",       # auto-detect (image, video, etc.)
                folder="my_app_uploads",    # optional but important
                overwrite=True
            )

        result = await loop.run_in_executor(None, _upload)
        return result["secure_url"]


      
   def uploadsync(self, data: bytes, filename: str) -> str:
        result = cloudinary.uploader.upload(
            BytesIO(data),
            public_id=filename,
            resource_type="auto",
            folder="my_app_uploads",
            overwrite=True
        )
        return result["secure_url"]