import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download

from pathlib import Path
from .base import InpaintingStrategy
from app.celery.model_registry import register_strategy, get_model
from app.Exceptions.Internal_error import InpaintingException


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
        x1 = box["offset_x"]
        y1 = box["offset_y"]
        x2 = x1 + int(box["width"])
        y2 = y1 + int(box["height"])
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

    @staticmethod
    def inpaint_all_polygons(img_np, polygons):
        h, w = img_np.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        for polygon in polygons:
            poly = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(combined_mask, [poly], 255)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

        img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(combined_mask)
        return img_pil, mask_pil

    @staticmethod
    def lama_inpaint_batch(model, device, img_mask_pairs: list[tuple]) -> list:
        """Run LaMa once on a batch of (img_pil, mask_pil) pairs, padded to a shared size."""
        img_arrs, msk_arrs, original_sizes = [], [], []

        for img_pil, mask_pil in img_mask_pairs:
            img_np = np.array(img_pil.convert("RGB")).astype(np.float32) / 255.0
            msk_np = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0
            original_sizes.append((img_np.shape[0], img_np.shape[1]))
            img_arrs.append(img_np)
            msk_arrs.append(msk_np)

        max_h = max(h for h, w in original_sizes)
        max_w = max(w for h, w in original_sizes)
        pad_h = (8 - max_h % 8) % 8
        pad_w = (8 - max_w % 8) % 8
        target_h, target_w = max_h + pad_h, max_w + pad_w

        img_tensors, msk_tensors = [], []
        for img_np, msk_np, (h, w) in zip(img_arrs, msk_arrs, original_sizes):
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)   # (C, H, W)
            msk_t = torch.from_numpy(msk_np).unsqueeze(0)        # (1, H, W)

            pad_bottom, pad_right = target_h - h, target_w - w
            img_t = F.pad(img_t.unsqueeze(0), (0, pad_right, 0, pad_bottom), mode="reflect").squeeze(0)
            msk_t = F.pad(msk_t.unsqueeze(0), (0, pad_right, 0, pad_bottom), mode="reflect").squeeze(0)

            img_tensors.append(img_t)
            msk_tensors.append(msk_t)

        img_batch = torch.stack(img_tensors).to(device)  # (N, C, target_h, target_w)
        msk_batch = torch.stack(msk_tensors).to(device)  # (N, 1, target_h, target_w)

        try:
            with torch.no_grad():
                result_batch = model(img_batch, msk_batch)
        except Exception as e:
            raise InpaintingException("LaMa batch inpainting failed", stage="inpainting") from e

        outputs = []
        for i, (h, w) in enumerate(original_sizes):
            result_t = result_batch[i:i + 1, :, :h, :w]
            result_np = result_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result_np = np.clip(result_np * 255, 0, 255).astype(np.uint8)
            outputs.append(Image.fromarray(result_np))

        return outputs

    def process_image(self, images_boxes: list[list[dict]], images: list) -> list:
        pairs = []          # (img_pil, mask_pil) only for pages that actually have boxes
        pair_page_idx = []  # maps each pair back to its position in `images`
        results = list(images)  # pages without boxes pass through untouched

        for idx, page_boxes in enumerate(images_boxes):
            if not page_boxes:
                continue
            page_img = images[idx]
            polygons = [self._box_to_polygon(box) for box in page_boxes]
            img_pil, mask_pil = self.inpaint_all_polygons(page_img, polygons)
            pairs.append((img_pil, mask_pil))
            pair_page_idx.append(idx)

        if not pairs:
            return results  # nothing needed inpainting this call

        inpainted = self.lama_inpaint_batch(self.model, self.device, pairs)

        for pair_pos, page_idx in enumerate(pair_page_idx):
            h, w = images[page_idx].shape[:2]
            result_np = cv2.cvtColor(np.array(inpainted[pair_pos]), cv2.COLOR_RGB2BGR)
            if result_np.shape[:2] != (h, w):
                result_np = cv2.resize(result_np, (w, h))
            results[page_idx] = result_np

        return results