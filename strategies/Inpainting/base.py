import base64
import aiohttp
import cv2
import numpy as np
from abc import ABC, abstractmethod

from app.config import settings   #  correct import

# return the text in the image given the bounding box of the text in the image
class InpaintingStrategy(ABC):
    @abstractmethod
    async def process_image(self, self, images_boxes: list[list[dict]], images: list) -> list
        pass
