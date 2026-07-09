import cv2
import numpy as np
import cairo
import gi
gi.require_version('Pango', '1.0')
gi.require_version('PangoCairo', '1.0')
from gi.repository import Pango, PangoCairo

from app.exceptions import RenderingException

import logging
logger = logging.getLogger(__name__)


def _clear_pango_cache():
    PangoCairo.font_map_get_default().changed()


def find_best_font(text: str, w: int, h: int, base_font: str = "Sans Bold") -> int:
    try:
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        layout = PangoCairo.create_layout(ctx)
        layout.set_text(text, -1)

        lo, hi = 6, 80
        best_size = lo

        while lo <= hi:
            mid = (lo + hi) // 2
            font_desc = Pango.FontDescription(f"{base_font} {mid}")
            layout.set_font_description(font_desc)
            layout.set_width(w * Pango.SCALE)
            layout.set_wrap(Pango.WrapMode.WORD)

            _, text_h = layout.get_pixel_size()
            if text_h <= h:
                best_size = mid
                lo = mid + 1
            else:
                hi = mid - 1

        surface.finish()
        return best_size
    except Exception as e:
        raise RenderingException("Font size search failed", stage="rendering") from e


def get_pango_text_image(text: str, w: int, h: int) -> np.ndarray:
    try:
        font_family = "Sans Bold"
        best_size = find_best_font(text, w, h, base_font=font_family)  # already raises RenderingException

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)

        ctx.set_source_rgba(0, 0, 0, 0)
        ctx.paint()

        layout = PangoCairo.create_layout(ctx)
        font_desc = Pango.FontDescription(f"{font_family} {best_size}")
        layout.set_font_description(font_desc)
        layout.set_text(text, -1)
        layout.set_width(w * Pango.SCALE)
        layout.set_wrap(Pango.WrapMode.WORD)
        layout.set_alignment(Pango.Alignment.CENTER)

        _, text_h = layout.get_pixel_size()
        start_y = max(0, (h - text_h) // 2)

        ctx.move_to(0, start_y)
        ctx.set_source_rgb(0, 0, 0)
        PangoCairo.show_layout(ctx, layout)

        buf = surface.get_data()
        render = np.ndarray(shape=(h, w, 4), dtype=np.uint8, buffer=buf)
        result = render.copy()

        surface.finish()
        _clear_pango_cache()

        return result
    except RenderingException:
        raise  
    except Exception as e:
        raise RenderingException("Pango text image generation failed", stage="rendering") from e


def overlay_arabic_text(img: np.ndarray, polygon: list, text: str) -> np.ndarray:
    poly = np.array(polygon, dtype=np.int32)
    x1, y1, w, h = cv2.boundingRect(poly)
    x2, y2 = x1 + w, y1 + h

    if w <= 0 or h <= 0:
        return img

    text_overlay = get_pango_text_image(text, w, h)  # already raises RenderingException

    try:
        text_bgr = text_overlay[:, :, :3]
        alpha_mask = text_overlay[:, :, 3] / 255.0

        roi = img[y1:y2, x1:x2]
        if roi.shape[:2] != (h, w):
            raise ValueError(f"ROI shape {roi.shape[:2]} doesn't match expected ({h}, {w}) — box out of image bounds")

        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + text_bgr[:, :, c] * alpha_mask
        img[y1:y2, x1:x2] = roi
    except Exception as e:
        raise RenderingException("Overlay compositing failed", stage="rendering") from e

    return img


def render_translated_image(images: list, data: list) -> list:
    for idx, page in enumerate(data):
        if len(page) == 0:
            continue
        for aoi in page:
            text = aoi.get("text", "").strip()
            if not text:
                continue

            x, y = aoi.get("offset_x"), aoi.get("offset_y")
            w, h = aoi.get("width"), aoi.get("height")
            if None in (x, y, w, h):
                logger.warning(
                    "Skipping box with missing coordinates",
                    extra={"page_idx": idx, "box_index": aoi.get("box_index")},
                )
                continue

            polygon = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

            try:
                images[idx] = overlay_arabic_text(images[idx], polygon, text)
            except RenderingException as e:
                logger.warning(
                    f"Skipping box {aoi.get('box_index')} on page {idx}: {e}",
                    extra={"page_idx": idx, "box_index": aoi.get("box_index")},
                )
                continue  # keep rendering the rest of the page rather than failing the whole chapter

    return images