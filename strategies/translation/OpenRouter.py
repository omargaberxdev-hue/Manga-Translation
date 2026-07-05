from typing import List

import requests

from .base import TranslationStrategy
from app.config import settings
from app.celery.model_registry import register_strategy

@register_strategy
class OpenRouterTranslation(TranslationStrategy):
    SYSTEM_PROMPT = (
        "You are a professional translation engine. You will receive a JSON "
        "array of source sentences. Respond with ONLY a valid JSON array of "
        "the translated strings, in the exact same order and count as the "
        "input. Do not include markdown code fences, explanations, labels, "
        "or any text outside the JSON array itself. Your entire response "
        "must be parseable directly by json.loads()."
    )


    def __init__(self, to_lang: str = "en"):
        self.to_lang = to_lang

    import json
from typing import List

import requests

from .base import TranslationStrategy
from app.config import settings
from app.celery.model_registry import register_strategy


import json
from typing import List

import requests

from .base import TranslationStrategy
from app.config import settings
from app.celery.model_registry import register_strategy


@register_strategy
class OpenRouterTranslation(TranslationStrategy):

    SYSTEM_PROMPT = (
        "You are a translation engine. You will receive numbered source "
        "sentences. Respond with ONLY a JSON object mapping each number to "
        "its translation, like {\"1\": \"translated text\", \"2\": \"translated text\"}. "
        "No markdown, no explanations, nothing outside the JSON object."
    )

    def __init__(self, to_lang: str = "en"):
        self.to_lang = to_lang

    def Translate(self, text: List[str]) -> List[str]:
        numbered_input = {str(i + 1): sentence for i, sentence in enumerate(text)}

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openrouterapikey}",
                "Content-Type": "application/json",
            },
            json={
                "model": "poolside/laguna-xs.2:free",
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Target language: {self.to_lang}\n"
                            f"{json.dumps(numbered_input)}"
                        ),
                    },
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        if response.status_code >= 400:
           print(f"OpenRouter error body: {response.text}")
        response.raise_for_status()
        

        message_content = response.json()["choices"][0]["message"]["content"]
        result = json.loads(message_content)

        return [result[str(i + 1)] for i in range(len(text))]

    def translate_blocks(self, pages: List[dict]) -> List[dict]:
        flat_text = []
        page_lengths = []

        for page in pages:
            valid_boxes = [box for box in page if box.get("ocr_text") is not None]
            page_lengths.append(len(valid_boxes))
            flat_text.extend(box["ocr_text"] for box in valid_boxes)

        if not flat_text:
            return [[] for _ in pages]

        translated = self.Translate(flat_text)

        all_results = []
        cursor = 0
        for page, length in zip(pages, page_lengths):
            valid_boxes = [box for box in page if box.get("ocr_text") is not None]

            page_translation_results = translated[cursor:cursor + length]
            cursor += length

            page_results = []
            for box_meta, text in zip(valid_boxes, page_translation_results):
                page_results.append({
                    "page_id": box_meta["page_id"],
                    "manga_id": box_meta["manga_id"],
                    "box_index": box_meta["box_index"],
                    "offset_x": box_meta["offset_x"],
                    "offset_y": box_meta["offset_y"],
                    "width": box_meta["width"],   # was "weidth" — typo fixed, verify this matches your box schema
                    "height": box_meta["height"],
                    "confidence": box_meta["confidence"],
                    "text": text,
                })

            all_results.append(page_results)

        return all_results