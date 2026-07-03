from paddleocr import PaddleOCR as _PaddleOCR

from .base import OCRStrategy
from app.celery.model_registry import register_strategy
@register_strategy
class PaddleOCRStrategy(OCRStrategy):

    @staticmethod
    def load_model(use_gpu=True):
    
        ocr = _PaddleOCR(
            det=False,
            rec=True,
            cls=True,
            lang='japan',
            use_gpu=use_gpu,
            show_log=False
        )
        return ocr

    def __init__(self, use_gpu=True):
        self.ocr = get_model("extraction")

    def extract(self, pages):
        flat_images = []    
        page_lengths = []    

        for page in pages:
            valid_boxes = [box for box in page if box.get("image") is not None]
            page_lengths.append(len(valid_boxes))
            flat_images.extend(box["image"] for box in valid_boxes)

        if not flat_images:
            return [[] for _ in pages]

        ocr_output = self.ocr.ocr(flat_images, cls=True)


        all_results = []
        cursor = 0
        for page, length in zip(pages, page_lengths):
            valid_boxes = [box for box in page if box.get("image") is not None]
            
            page_ocr_results = ocr_output[cursor:cursor + length]
            cursor += length

            page_results = []
            for box_meta, ocr_result in zip(valid_boxes, page_ocr_results):
                page_results.append({
                    "page_id": box_meta["page_id"],
                    "manga_id": box_meta["manga_id"],
                    "box_index": box_meta["box_index"],
                    "offset_x": box_meta["offset_x"],
                    "offset_y": box_meta["offset_y"],
                    "width": box_meta["weidth"],   
                    "height": box_meta["height"],
                    "confidence": box_meta["confidence"],
                    "ocr_text": ocr_result,      
                })

            all_results.append(page_results)

        return all_results