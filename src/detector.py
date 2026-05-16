"""
Garment detection + cropping using trained YOLOv8 model.

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image
from ultralytics import YOLO


# Trained class names (DeepFashion2 13-class mapping)
CLASS_NAMES: List[str] = [
    "short_sleeve_top",
    "long_sleeve_top",
    "short_sleeve_outwear",
    "long_sleeve_outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short_sleeve_dress",
    "long_sleeve_dress",
    "vest_dress",
    "sling_dress",
]


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class DetectionItem:
    image_path: str
    crop_path: str
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]  # [x1,y1,x2,y2]


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _safe_mkdir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXTS


def _collect_sources(source: Union[str, Path]) -> List[Path]:
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    if src.is_file():
        if not _is_image_file(src):
            raise ValueError(f"Not a supported image file: {src}")
        return [src]

    # directory
    files = [p for p in src.rglob("*") if _is_image_file(p)]
    if not files:
        raise RuntimeError(f"No images found in folder: {src}")
    return sorted(files)


def crop_with_margin(img: Image.Image, xyxy: Sequence[float], margin: float) -> Image.Image:
    """Crop with a small margin around bbox to avoid cutting garment edges."""
    w, h = img.size
    x1, y1, x2, y2 = map(float, xyxy)

    bw = x2 - x1
    bh = y2 - y1
    x1 -= margin * bw
    y1 -= margin * bh
    x2 += margin * bw
    y2 += margin * bh

    x1i = _clamp(int(round(x1)), 0, w - 1)
    y1i = _clamp(int(round(y1)), 0, h - 1)
    x2i = _clamp(int(round(x2)), 1, w)
    y2i = _clamp(int(round(y2)), 1, h)

    if x2i <= x1i or y2i <= y1i:
        # fallback
        return img.copy()

    return img.crop((x1i, y1i, x2i, y2i))


class GarmentDetector:
    def __init__(
        self,
        model_path: Union[str, Path],
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 512,
        margin: float = 0.0,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.margin = float(margin)
        self.class_names = class_names or CLASS_NAMES

        self.model = YOLO(str(self.model_path))

    def detect_and_crop(
        self,
        image_path: Union[str, Path],
        out_dir: Union[str, Path],
        max_crops: Optional[int] = None,
    ) -> List[DetectionItem]:
        image_path = Path(image_path)
        out_dir = Path(out_dir)
        _safe_mkdir(out_dir)

        img = Image.open(image_path).convert("RGB")

        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # Sort detections by confidence (high -> low)
        confs = boxes.conf.cpu().numpy()
        order = confs.argsort()[::-1]

        if max_crops is not None:
            order = order[: int(max_crops)]

        stem = image_path.stem
        items: List[DetectionItem] = []

        for rank, idx in enumerate(order):
            b = boxes[int(idx)]
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = b.xyxy[0].tolist()

            cls_name = (
                self.class_names[cls_id]
                if 0 <= cls_id < len(self.class_names)
                else f"class_{cls_id}"
            )

            crop = crop_with_margin(img, (x1, y1, x2, y2), margin=self.margin)

            # Safe filename (no weird chars)
            crop_name = f"{stem}_{rank:02d}_{cls_name}_{conf:.2f}.jpg"
            crop_path = out_dir / crop_name
            crop.save(crop_path, quality=95)

            items.append(
                DetectionItem(
                    image_path=str(image_path),
                    crop_path=str(crop_path),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                )
            )

        return items


def _default_model_path() -> Path:
    
    here = Path(__file__).resolve()
    project_root = here.parents[1]
    return project_root / "models" / "best.pt"


def _default_out_dir() -> Path:
    here = Path(__file__).resolve()
    project_root = here.parents[1]
    return project_root / "data" / "crops"


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO garment detection + cropper")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image file OR a folder containing images",
    )
    parser.add_argument(
        "--model",
        default=str(_default_model_path()),
        help="Path to trained YOLO weights (best.pt). Default: models/best.pt",
    )
    parser.add_argument(
        "--out",
        default=str(_default_out_dir()),
        help="Output folder for cropped garment images. Default: data/crops",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size")
    parser.add_argument("--margin", type=float, default=0.05, help="Crop margin ratio")
    parser.add_argument(
        "--max-crops",
        type=int,
        default=None,
        help="Limit number of crops per image (highest confidence first)",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save a crops_metadata.json file into the output folder",
    )

    args = parser.parse_args()

    sources = _collect_sources(args.source)
    out_dir = Path(args.out)
    _safe_mkdir(out_dir)

    detector = GarmentDetector(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        margin=args.margin,
        class_names=CLASS_NAMES,
    )

    all_items: List[DetectionItem] = []

    for img_path in sources:
        items = detector.detect_and_crop(img_path, out_dir, max_crops=args.max_crops)
        all_items.extend(items)
        print(f"{img_path.name}: {len(items)} crop(s)")

    print(f"\nTotal crops saved: {len(all_items)}")
    print(f"Output folder: {out_dir}")

    if args.save_json:
        meta_path = out_dir / "crops_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([asdict(x) for x in all_items], f, indent=2)
        print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
