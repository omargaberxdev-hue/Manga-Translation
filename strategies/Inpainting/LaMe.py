import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

from pathlib import Path
from .base import InpaintingStrategy
from app.celery.model_registry import register_strategy ,get_model

@register_strategy
class LaMeInpainting(InpaintingStrategy):

    @staticmethod
    def load_model(device: str = "cpu"):
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_weight_path = models_dir / "big-lama.pt"
        if not model_weight_path.exists():
            hf_hub_download(
                repo_id="fashn-ai/LaMa",
                filename="big-lama.pt",
                local_dir=str(models_dir),
                local_dir_use_symlinks=False
            )
        
        model = torch.jit.load(model_weight_path, map_location=device)
        model.eval()
        print(f"LaMa loaded ({device})")
        return model
    @classmethod
    def share_memory(cls, model):
        model.share_memory()

    def __init__(self):
        self.model = get_model("inpaint")
        self.device = get_model("device")

    @staticmethod
    def _box_to_polygon(box: dict) -> np.ndarray:
        """Convert offset_x/y + width/height dict to a 4-point polygon."""
        x1 = box["offset_x"]
        y1 = box["offset_y"]
        x2 = x1 + int(box["width"])
        y2 = y1 + int(box["height"])
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

    @staticmethod
    def lama_inpaint(model, device, img_pil, mask_pil):
        img_np = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
        msk_np = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0

        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        msk_t = torch.from_numpy(msk_np).unsqueeze(0).unsqueeze(0).to(device)

        h, w  = img_t.shape[2], img_t.shape[3]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
        msk_t = F.pad(msk_t, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            result_t = model(img_t, msk_t)

        result_t  = result_t[:, :, :h, :w]
        result_np = result_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result_np)

    @staticmethod
    def inpaint_all_polygons(lama_model, device, img_np, polygons):
        h, w          = img_np.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        kernel        = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        for polygon in polygons:
            poly = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(combined_mask, [poly], 255)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        img_pil    = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        mask_pil   = Image.fromarray(combined_mask)
        result_pil = LaMeInpainting.lama_inpaint(lama_model, device, img_pil, mask_pil)
        result_np  = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

        if result_np.shape[:2] != (h, w):
            result_np = cv2.resize(result_np, (w, h))

        return result_np

    def process_image(self, images_boxes: list[list[dict]], images: list) -> list:
      results = []
      for idx, page_boxes in enumerate(images_boxes):
            page_img = images[idx]
            if not page_boxes:
                results.append(page_img)
                continue
            polygons = [self._box_to_polygon(box) for box in page_boxes]
            inpainted = self.inpaint_all_polygons(self.model, self.device, page_img, polygons)
            results.append(inpainted)
      return results