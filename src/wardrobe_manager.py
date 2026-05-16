"""
Wardrobe Manager
To run from project root:
  python -m src.wardrobe_manager add --image "data/test_images/img1.jpg" --category "trousers"
  python -m src.wardrobe_manager list
  python -m src.wardrobe_manager remove --item-id "<id>"
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.detector import GarmentDetector, DetectionItem, CLASS_NAMES
from src.fashionclip_embedder import FashionCLIPEmbedder
from src.color_detector import detect_dominant_color
from src.zara_mapper import suggest_zara_categories


# Default Paths
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _wardrobe_dir() -> Path:
    return _project_root() / "data" / "wardrobe"


def _original_dir() -> Path:
    return _wardrobe_dir() / "original"


def _crops_dir() -> Path:
    return _wardrobe_dir() / "crops"


def _embeddings_dir() -> Path:
    return _wardrobe_dir() / "embeddings"


def _db_path() -> Path:
    return _wardrobe_dir() / "wardrobe.json"


def _model_path() -> Path:
    return _project_root() / "models" / "best.pt"


# Wardrobe item record
@dataclass
class WardrobeItem:
    item_id: str
    original_path: str
    crop_path: str
    embedding_path: str
    zara_category: str
    yolo_class: str
    yolo_confidence: float
    color: str
    added_at: str  # ISO datetime

# Utility functions
def _safe_mkdir(p: Path) -> None:   # Creates directories safely
    p.mkdir(parents=True, exist_ok=True)


def _load_db() -> List[Dict[str, Any]]:   # Loads wardrobe JSON database.
    path = _db_path()
    if not path.exists():
        return []

    if path.stat().st_size == 0:
        return []

    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save_db(records: List[Dict[str, Any]]) -> None:
    path = _db_path()
    _safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _copy_image_to_folder(image_path: Path, dest_folder: Path, item_id: str) -> Path:
    _safe_mkdir(dest_folder)
    ext = image_path.suffix.lower() if image_path.suffix else ".jpg"
    dest = dest_folder / f"{item_id}{ext}"
    shutil.copy2(image_path, dest)
    return dest


def _save_embedding(vec: np.ndarray, dest_folder: Path, item_id: str) -> Path:
    _safe_mkdir(dest_folder)
    dest = dest_folder / f"{item_id}.npy"
    np.save(dest, vec)
    return dest

# Zara -> YOLO mapping (for crop selection)
ZARA_TO_YOLO: Dict[str, set[str]] = {
    # Tops
    "top": {"short_sleeve_top", "long_sleeve_top", "vest", "sling"},
    "t-shirt": {"short_sleeve_top", "long_sleeve_top"},
    "shirt": {"long_sleeve_top", "short_sleeve_top"},
    "knitwear": {"long_sleeve_top", "vest"},
    "cardigan": {"long_sleeve_outwear"},
    "sweatshirt": {"long_sleeve_top", "short_sleeve_top"},
    "vest": {"vest"},

    # Bottoms
    "trousers": {"trousers"},
    "jeans": {"trousers"},
    "skirt": {"skirt"},
    "shorts": {"shorts"},

    # Dresses
    "dress": {"short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"},

    # Outerwear
    "jacket": {"short_sleeve_outwear", "long_sleeve_outwear"},
    "coat": {"long_sleeve_outwear"},
    "blazer": {"long_sleeve_outwear"},
}


# choose the best crop from multiple detections matches the user category.
def _pick_best_crop_for_user_category(
    crops: List[DetectionItem],
    zara_category: str,
) -> Tuple[DetectionItem, str]:
    
    if not crops:
        raise RuntimeError("No crops to choose from.")

    cat = (zara_category or "").strip().lower()
    allowed = ZARA_TO_YOLO.get(cat, set())

    # Ensure crops are sorted by confidence (high -> low)
    crops_sorted = sorted(crops, key=lambda c: float(c.confidence), reverse=True)

    # Prefer crops whose class matches the user category
    if allowed:
        matching = [c for c in crops_sorted if (c.class_name or "").lower() in allowed]
        if matching:
            return matching[0], "matched_user_category"

    # If none match, fallback to highest-confidence crop
    return crops_sorted[0], "fallback_highest_conf"


class WardrobeManager:
    def __init__(
        self,
        model_path: Path | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 512,
        margin: float = 0.05,
    ) -> None:
        self.model_path = model_path or _model_path()

        # YOLO cropper
        self.detector = GarmentDetector(
            model_path=self.model_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            margin=margin,
            class_names=CLASS_NAMES,
        )

        # FashionCLIP embedder
        self.embedder = FashionCLIPEmbedder()

        # Ensure folders exist
        _safe_mkdir(_original_dir())
        _safe_mkdir(_crops_dir())
        _safe_mkdir(_embeddings_dir())

    # Adding a new clothing item into the digital wardrobe.
    def add_item(
        self,
        image_path: str | Path,
        zara_category: str,
        max_crops: int = 5,
    ) -> WardrobeItem:
        image_path = Path(image_path)
        if not image_path.exists():     # Validate image path
            raise FileNotFoundError(f"Image not found: {image_path}")

        item_id = uuid.uuid4().hex[:12]    # Generate unique item ID

        # Save original image
        saved_original = _copy_image_to_folder(image_path, _original_dir(), item_id)

        # Detect and crop
        crops: List[DetectionItem] = self.detector.detect_and_crop(
            saved_original,
            out_dir=_crops_dir(),
            max_crops=max_crops,
        )

        if not crops:
            raise RuntimeError(
                "No garment detected. Try another image or lower --conf in WardrobeManager init."
            )

        # Pick best crop that matches user chosen Zara category (fallback to highest confidence)
        best_crop, mode = _pick_best_crop_for_user_category(crops, zara_category)

        # AI Zara category suggestion 
        suggestions = suggest_zara_categories(best_crop.class_name)

        print("\n AI Category Suggestion")
        print(f"YOLO detected: {best_crop.class_name}")
        print(f"Primary suggestion: {suggestions['primary']}")
        print(f"Alternatives: {suggestions['alternatives']}")

        if mode != "matched_user_category":
            print(
                f"   Warning: YOLO crop did not match user category.\n"
                f"   user zara_category='{zara_category}'\n"
                f"   chosen yolo_class='{best_crop.class_name}' conf={best_crop.confidence:.2f}\n"
                f"   (mode={mode})"
            )

        # Detect colour 
        color = detect_dominant_color(best_crop.crop_path)

        # Generate embedding
        emb = self.embedder.get_image_embedding(best_crop.crop_path)

        # Save embedding
        saved_emb = _save_embedding(emb, _embeddings_dir(), item_id)

        # Create record
        record = WardrobeItem(
            item_id=item_id,
            original_path=str(saved_original),
            crop_path=best_crop.crop_path,
            embedding_path=str(saved_emb),
            zara_category=zara_category,
            yolo_class=best_crop.class_name,
            yolo_confidence=float(best_crop.confidence),
            color=color,
            added_at=_now_iso(),
        )

        # Save to database
        db = _load_db()
        db.append(asdict(record))
        _save_db(db)

        return record

    # Loads database and returns all records as WardrobeItem objects.
    def list_items(self) -> List[WardrobeItem]:
        return [WardrobeItem(**x) for x in _load_db()]

    # Deletes item from wardrobe
    def remove_item(self, item_id: str) -> bool:
        db = _load_db()
        new_db = [x for x in db if x.get("item_id") != item_id]
        if len(new_db) == len(db):
            return False

        for rec in db:
            if rec.get("item_id") == item_id:
                for key in ["original_path", "crop_path", "embedding_path"]:
                    p = Path(rec.get(key, ""))
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass

        _save_db(new_db)
        return True

# CLI (For testing and using the wardrobe manager independently from the web app.)
def main() -> None:
    parser = argparse.ArgumentParser(description="Wardrobe Manager CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add a wardrobe item")
    p_add.add_argument("--image", required=True, help="Path to item image")
    p_add.add_argument("--category", required=True, help="User-selected Zara category")
    p_add.add_argument("--max-crops", type=int, default=5, help="Max crops per item (pick best)")

    p_list = sub.add_parser("list", help="List wardrobe items")

    p_rm = sub.add_parser("remove", help="Remove a wardrobe item by ID")
    p_rm.add_argument("--item-id", required=True)

    args = parser.parse_args()

    wm = WardrobeManager()

    if args.cmd == "add":
        rec = wm.add_item(args.image, args.category, max_crops=args.max_crops)
        print(" Added item:")
        print(json.dumps(asdict(rec), indent=2))

    elif args.cmd == "list":
        items = wm.list_items()
        if not items:
            print("Wardrobe is empty.")
            return
        for it in items:
            print(
                f"- {it.item_id} | {it.zara_category} | "
                f"yolo={it.yolo_class} ({it.yolo_confidence:.2f}) | color={it.color}"
            )
            print(f"  crop: {it.crop_path}")

    elif args.cmd == "remove":
        ok = wm.remove_item(args.item_id)
        print(" Removed." if ok else "Item ID not found.")


if __name__ == "__main__":
    main()