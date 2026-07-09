import torch
from PIL import Image

from .base import OCRStrategy

from app.celery.model_registry import register_strategy, get_model
from app.exceptions import OCRException


@register_strategy
class TrOCRStrategy(OCRStrategy):

    @staticmethod
    def load_model(use_gpu: bool = False):
        processor_instance = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model_instance = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        if use_gpu:
            model_instance = model_instance.to("cuda")

        return model_instance, processor_instance

    def __init__(self, Lang=None):
        self.use_gpu = (get_model("device") == "cuda")
        self.model_instance, self.processor_instance = get_model("extraction")

    def extract(self, pages):
        flat_image = []
        page_lengths = []

        for page in pages:
            valid_boxes = [box for box in page if box.get("image") is not None]
            page_lengths.append(len(valid_boxes))
            flat_image.extend(box["image"] for box in valid_boxes)

        if not flat_image:
            return [[] for _ in pages]

        try:
            images = [Image.fromarray(img).convert("RGB") for img in flat_image]

            pixel_values = self.processor_instance(images, return_tensors="pt").pixel_values
            if self.use_gpu:
                pixel_values = pixel_values.to("cuda")

            generated_ids = self.model_instance.generate(pixel_values, max_new_tokens=32)
            texts = self.processor_instance.batch_decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            raise OCRException("TrOCR inference failed", stage="ocr") from e

        all_results = []
        cursor = 0
        for page, length in zip(pages, page_lengths):
            valid_boxes = [box for box in page if box.get("image") is not None]

            page_extraction_results = texts[cursor:cursor + length]
            cursor += length

            page_results = []
            for box_meta, text in zip(valid_boxes, page_extraction_results):
                page_results.append({
                    "page_id": box_meta["page_id"],
                    "manga_id": box_meta["manga_id"],
                    "box_index": box_meta["box_index"],
                    "offset_x": box_meta["offset_x"],
                    "offset_y": box_meta["offset_y"],
                    "width": box_meta["width"],
                    "height": box_meta["height"],
                    "confidence": box_meta["confidence"],
                    "ocr_text": text,
                })

            all_results.append(page_results)

        return all_results