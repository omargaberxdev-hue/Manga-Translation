from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from app.utils.Lame import Inpainting
from app.utils.image_utils import download_image


from .base import DetectionStrategy


from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future

from app.celery.model_registry import register_strategy , get_model


# [{
#  "MangaID": "string",
#  "CanvasURL": "string",
#  "Chapters_data": {
#     "ChapterID": "string",
#     "Pages": [
#        {
#          "PageID": "string",
#          "x1": 0,
#          "y1": 0,
#          "w": 0,
#          "h": 0
#        }
#     ]
#  }
# }]

@register_strategy
class TextSegmenter(DetectionStrategy):
    
    @staticmethod
    def load_model():
        models_dir = Path(__file__).parent.parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_weight_path = models_dir / "comic-text-segmenter.pt"
        if not model_weight_path.exists():
            hf_hub_download(                         
                repo_id="ogkalu/comic-text-segmenter-yolov8m",
                filename="comic-text-segmenter.pt",
                local_dir=str(models_dir),
                local_dir_use_symlinks=False
            )
        
        return YOLO(model_weight_path)                #

    
    @classmethod
    def share_memory(cls, model):
        model.model.share_memory()


    def __init__(self):
        self.model = get_model(settings.detection_strategy)
        self.executor = ThreadPoolExecutor()  


    def detect(self, data) -> tuple[list, Future[list]]:
        manga_id = data['manga_id']

        image_url =  data['canvas_url']

        chapter_data = data['chapter_data']

        pages =  data["pages"]

        all_boxes = []

        # Load chapter canvas
        img =  download_image(image_url)
        canvas_np = np.array(img)
   
           
         # Build page crops and metadata
        page_images = []
        page_metas = []

        for page in pages:
            page_id = page['page_id']

            x1, y1, w, h = page['x1'], page['y1'], page['w'], page['h']
            page_crop = canvas_np[y1:y1+h, x1:x1+w]
                
            page_images.append(page_crop)
            page_metas.append({
                    "page_id": page['page_id'],
                    "image" : page_crop,
                    "page_x1": x1,
                    "page_y1": y1,
                    "manga_id": manga_id
                })

        # Batch detect all pages in this chapter
        results = self.model(page_images, imgsz=1024, verbose=False)
            
        # zip aligns: results[i] -> page_images[i] -> page_metas[i]
        for result, page_img, meta in zip(results, page_images, page_metas):
            page_boxes = []
            
            if result.boxes is None or len(result.boxes) == 0:
                all_boxes.append(page_boxes)
                continue

            boxes_xywh = result.boxes.xywh.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes_xywh, confs)):
                cx, cy, bw, bh = box
               
                # Crop AOI from the page image using relative box coords
                rx1 = int(cx - bw / 2)
                ry1 = int(cy - bh / 2)
                rx2 = int(cx + bw / 2)
                ry2 = int(cy + bh / 2)
                aoi_crop = page_img[ry1:ry2, rx1:rx2]
                page_boxes.append({
                    "page_id": meta['page_id'],
                    "manga_id": meta['manga_id'],
                    "box_index": i,
                    "offset_x": rx1,
                    "offset_y": ry1,
                    "weidth": bw,
                    "height": bh,
                    "confidence": float(conf),
                    "image": aoi_crop

                })

            all_boxes.append(page_boxes)  
          
        

        inpaint = Inpainting()
        # in detect():
        future = self.executor.submit(inpaint.process_image_with_lama, all_boxes, page_images)


        return (all_boxes , future )
