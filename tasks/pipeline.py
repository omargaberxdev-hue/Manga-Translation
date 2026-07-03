# app/tasks/pipeline.py
import json
import logging

import cv2
import numpy as np

from app.celery.celery import celery
from app.config import settings
from app.utils.cache import ImageCache
from app.utils.renderer import render_translated_image

from app.strategies.cdn.strategy import get_cdn_strategy
from app.strategies.detection.factory import get_detection_stratgy
from app.strategies.ocr.strategy import get_ocr_strategy
from app.strategies.translation.strategy import get_translation_stratgy

logger = logging.getLogger(__name__)


def _glue_pages(page_ids: list[str], images: list[np.ndarray]) -> tuple[np.ndarray, list[dict]]:
    """Stitch translated page images side-by-side into a single canvas."""
    height = 0
    width = 0
    for image in images:
        h, w = np.array(image).shape[:2]
        height = max(h, height)
        width += w

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    pages_meta = []
    cursor = 0
    for order, (page_id, image) in enumerate(zip(page_ids, images)):
        h, w = np.array(image).shape[:2]
        canvas[0:h, cursor:cursor + w] = image
        pages_meta.append({
            "PageID": page_id,
            "x1": cursor,
            "y1": 0,
            "w": w,
            "h": h,
            "order": order,
        })
        cursor += w

    return canvas, pages_meta


#outbox

@celery.task(bind=True, max_retries=3, default_retry_delay=30)
def process(self, job: dict) -> dict:
    user_id = job["user_id"]
    comic_id = job["MangaID"]
    chapter_id = job["Chapters_data"]["ChapterID"]
    page_ids = [p["PageID"] for p in job["Chapters_data"]["Pages"]]

    # No None-check needed here: each factory either returns a ready
    # strategy or raises (e.g. HunyuanTranslation.__init__ raises
    # RuntimeError if build_pipeline() hasn't run yet for this process).
    detector = get_detection_stratgy(settings.detection_strategy)
    ocr_strategy = get_ocr_strategy(settings.ocr_strategy)
    cdn_strategy = get_cdn_strategy(settings.cdn_strategy)
    translation_strategy = get_translation_stratgy(settings.translation_strategy)

    try:
        boxes, inpaint_future = detector.detect(job)

        if not boxes:
            raise ValueError("No text regions detected")

        ocr_result = ocr_strategy.extract(boxes)

        after_translation = translation_strategy.translate_blocks(ocr_result)

        image_cleared = inpaint_future.result()

        final_images = render_translated_image(image_cleared, after_translation)

        if len(final_images) != len(page_ids):
            raise ValueError(
                f"Page count mismatch: {len(final_images)} rendered images "
                f"vs {len(page_ids)} page_ids"
            )

        canvas, pages_meta = _glue_pages(page_ids, final_images)

        _, img_encoded = cv2.imencode(".png", canvas)
        img_bytes = img_encoded.tobytes()
        cdn_url = cdn_strategy.upload(img_bytes, f"{chapter_id}_translated.png")  # local var, not worker.x

        cache = ImageCache()
        cache.redis.xadd(f"notifications:{user_id}", {
            "user_id": user_id,
            "status": "success",
            "pages_meta": json.dumps(pages_meta),
            "image_url": cdn_url,
            "comic_id": comic_id,
            "chapter_id": chapter_id,
            "cached": "false",
        })

        return {"status": "success", "cdn_url": cdn_url, "pages_meta": pages_meta}

    except Exception as exc:
        logger.exception(
            "Pipeline failed for user_id=%s comic_id=%s chapter_id=%s",
            user_id, comic_id, chapter_id,
        )

        try:
            cache = ImageCache()
            cache.redis.xadd(f"notifications:{user_id}", {
                "user_id": user_id,
                "status": "failed",
                "error": str(exc),
                "comic_id": comic_id,
                "chapter_id": chapter_id,
            })
        except Exception:
            logger.exception("Also failed to publish failure notification")

        raise self.retry(exc=exc)