# from paddleocr import PaddleOCR as _PaddleOCR

# from .base import OCRStrategy
# from app.celery.model_registry import register_strategy, get_model


# @register_strategy
# class PaddleOCRStrategy(OCRStrategy):

#     @staticmethod
#     def UseModel(Lang, use_gpu=True):
#         ocr = _PaddleOCR(
#             use_doc_orientation_classify=False,
#             use_doc_unwarping=False,
#             use_textline_orientation=False,
#             lang=Lang,
#             device="gpu" if use_gpu else "cpu",
#             engine="transformers"
#         )
#         return ocr
#     def __init__(self, Lang):
#         use_gpu = (get_model("device") == "cuda")
#         self.ocr = PaddleOCRStrategy.UseModel(Lang, use_gpu)

#     def extract(self, pages):
#         print("we start extracting ")
#         flat_images = []
#         page_lengths = []

#         for page in pages:
#             valid_boxes = [box for box in page if box.get("image") is not None]
#             page_lengths.append(len(valid_boxes))
#             flat_images.extend(box["image"] for box in valid_boxes)

#         if not flat_images:
#             return [[] for _ in pages]

#         ocr_output = list(self.ocr.predict(flat_images))

#         all_results = []
#         cursor = 0
#         for page, length in zip(pages, page_lengths):
#             valid_boxes = [box for box in page if box.get("image") is not None]
#             page_ocr_results = ocr_output[cursor:cursor + length]
#             cursor += length

#             page_results = []
#             for box_meta, res in zip(valid_boxes, page_ocr_results):
#                 text = "\n".join(res["rec_texts"]) if res["rec_texts"] else ""
#                 score = res["rec_scores"][0] if res["rec_scores"] else 0.0

#                 page_results.append({
#                     "page_id": box_meta["page_id"],
#                     "manga_id": box_meta["manga_id"],
#                     "box_index": box_meta["box_index"],
#                     "offset_x": box_meta["offset_x"],
#                     "offset_y": box_meta["offset_y"],
#                     "width": box_meta["width"],
#                     "height": box_meta["height"],
#                     "confidence": score,
#                     "ocr_text": text,
#                 })

#             all_results.append(page_results)

#         return all_results