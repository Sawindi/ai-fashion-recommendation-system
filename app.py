# app.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

from src.wardrobe_manager import WardrobeManager, _load_db
from src.zara_mapper import suggest_zara_categories, ZARA_CATEGORIES
from src.events_config import EVENTS
from src.outfit_generator import generate_outfits, complete_outfit


st.set_page_config(page_title="Fashion AI Wardrobe", layout="wide")

# Caching - creates the WardrobeManager only once and caches it.(Because it loads YOLO detector and FashionCLIP model. They are expensive to reload every time.)
@st.cache_resource
def get_manager() -> WardrobeManager:
    return WardrobeManager(conf=0.25, iou=0.5, imgsz=512, margin=0.05)


def pil_open_safe(path: str) -> Optional[Image.Image]:  # opens an image and converts it to RGB
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def format_item_card(rec: dict) -> str:  # creates a formatted text summary for a wardrobe item
    return (
        f"**{rec.get('zara_category', '?')}**  |  "
        f"YOLO: `{rec.get('yolo_class', '?')}` ({float(rec.get('yolo_confidence', 0.0)):.2f})  |  "
        f"Color: `{rec.get('color', '')}`  |  "
        f"ID: `{rec.get('item_id', '')}`"
    )


# UI
st.title("👗 Fashion AI Wardrobe Demo")

tabs = st.tabs(["➕ Add Item", "🧺 Wardrobe", "✨ Outfit Generator"])


# Tab 1: Add item
with tabs[0]:
    st.subheader("Add a wardrobe item")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    colA, colB = st.columns([1, 1])

    if uploaded is not None:     # Save upload temporarily
        tmp_dir = Path("data") / "tmp_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / uploaded.name
        tmp_path.write_bytes(uploaded.getbuffer())

        with colA:
            st.markdown("### Original image")
            st.image(str(tmp_path), use_container_width=True)  # Show original image

        # Load manager
        wm = get_manager()

        st.markdown("### 1) Detect garments (YOLO)")
        if st.button("Run detection", type="primary"):   # Run detection button
            preview_dir = Path("data") / "wardrobe" / "crops_preview"
            preview_dir.mkdir(parents=True, exist_ok=True)

            crops = wm.detector.detect_and_crop(
                image_path=tmp_path,
                out_dir=preview_dir,
                max_crops=8,
            )
            # Store crops in session state
            st.session_state["last_crops"] = crops
            st.session_state["last_upload_path"] = str(tmp_path)

        crops = st.session_state.get("last_crops", [])
        if crops:
            st.markdown("### 2) Crops found")
            grid_cols = st.columns(4)
            for i, c in enumerate(crops[:8]):   # Show detected crops
                img = pil_open_safe(c.crop_path)
                with grid_cols[i % 4]:
                    if img:
                        st.image(
                            img,
                            caption=f"{i}: {c.class_name} ({c.confidence:.2f})",
                            use_container_width=True,
                        )

            st.markdown("### 3) Choose crop")
            crop_index = st.number_input(     # Choose crop manually
                "Crop index (0 = best by confidence)",
                min_value=0,
                max_value=max(0, len(crops) - 1),
                value=0,
                step=1,
            )
            chosen_crop = crops[int(crop_index)]

            sugg = suggest_zara_categories(chosen_crop.class_name)
            st.markdown("### 4) Zara category")
            st.write(f"YOLO detected: `{chosen_crop.class_name}`")
            st.write(f"AI suggests: **{sugg['primary']}**")
            if sugg["alternatives"]:
                st.caption("Alternatives: " + ", ".join(sugg["alternatives"]))

            default_index = 0
            if sugg["primary"] in ZARA_CATEGORIES:
                default_index = ZARA_CATEGORIES.index(sugg["primary"])

            chosen_category = st.selectbox(   # Category confirmation
                "Confirm / change Zara category",
                options=ZARA_CATEGORIES,
                index=default_index,
            )

            st.markdown("### 5) Add to wardrobe database")
            if st.button("Add item to wardrobe", type="primary"):    # Add item to wardrobe
                rec = wm.add_item(
                    st.session_state["last_upload_path"],
                    chosen_category,
                    max_crops=8,
                )
                st.success(f"Added item {rec.item_id}")
                st.json(rec.__dict__)
                st.session_state["last_crops"] = []
        else:
            st.info("Upload an image, then click **Run detection** to preview crops.")


# Tab 2: Wardrobe
with tabs[1]:
    st.subheader("Wardrobe items")

    db = _load_db()
    if not db:
        st.info("Wardrobe is empty. Add items in the first tab.")
    else:
        db_sorted = sorted(db, key=lambda x: x.get("added_at", ""), reverse=True)  # Sort newest first

        cats = sorted(
            set((x.get("zara_category") or "").lower() for x in db_sorted if x.get("zara_category"))
        )
        filter_cat = st.selectbox("Filter by category", options=["(all)"] + cats, index=0)  # Filter by category

        shown = 0
        for rec in db_sorted:
            if filter_cat != "(all)" and (rec.get("zara_category") or "").lower() != filter_cat:
                continue

            shown += 1
            c1, c2 = st.columns([1, 2])

            with c1:
                img = pil_open_safe(rec.get("crop_path", ""))
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.caption("Crop image missing.")

            with c2:
                st.markdown(format_item_card(rec))
                st.caption(f"Added: {rec.get('added_at', '')}")
                st.caption(f"Crop path: {rec.get('crop_path', '')}")
                st.caption(f"Embedding: {rec.get('embedding_path', '')}")
                
                wm = get_manager()

                if st.button("🗑 Remove Item", key=f"remove_{rec['item_id']}"):
                    ok = wm.remove_item(rec["item_id"])
                    if ok:
                        st.success("Item removed successfully.")
                        st.rerun()
                    else:
                        st.error("Failed to remove item.")

            st.divider()

        st.caption(f"Showing {shown} item(s).")


# Tab 3: Outfit generator
with tabs[2]:
    st.subheader("Outfit generator")

    if not _load_db():
        st.info("Add some wardrobe items first.")
    else:
        mode = st.radio(   # Recommendation mode selector
            "Mode",
            ["Generate outfits for an event", "Complete outfit from an item"],
            horizontal=True,
        )

        event_keys = list(EVENTS.keys())
        event_key = st.selectbox("Event", options=event_keys, index=0)   # Event selection
        topk = st.slider("Top-K outfits", min_value=1, max_value=10, value=5)  # Let user choose number of recommendations

        
        # Mode 1: Generate outfits for an event
        if mode == "Generate outfits for an event":
            if st.button("Generate outfits", type="primary"):
                results = generate_outfits(event_key, top_k=topk)
                if not results:
                    st.warning("No outfits generated. Add more items (tops/bottoms/dresses/outerwear).")
                else:
                    for i, r in enumerate(results, start=1):
                        score = r["score"]
                        items = r["items"]

                        st.markdown(f"## Outfit #{i} — Score {score.total:.3f} ({'⭐' * score.rating_stars})")

                        cols = st.columns(len(items))
                        for j, it in enumerate(items):
                            with cols[j]:
                                img = pil_open_safe(it.get("crop_path", ""))
                                if img:
                                    st.image(img, use_container_width=True)
                                st.caption(f"{it['zara_category']} | {it['item_id']}")
                                st.caption(f"Color: {it.get('color', '')}")

                        with st.expander("Score breakdown & explanation"):
                            st.json({
                                "total": score.total,
                                "rating_stars": score.rating_stars,
                                "category_score": score.category_score,
                                "event_score": score.event_score,
                                "formality_score": score.formality_score,
                                "color_score": score.color_score,
                                "style_score": score.style_score,
                                "yolo_consistency_score": score.yolo_consistency_score,
                                "learned_score": score.learned_score,
                                "explanation": score.explanation,
                            })

                        st.divider()

        # Mode 2: Complete outfit from a base item
        else:
            db = _load_db()

            if "selected_base_id" not in st.session_state:   # Session state for selected base item
                st.session_state["selected_base_id"] = None

            st.markdown("### Choose a base garment")

            # image + button under each item
            preview_cols = st.columns(4)
            for i, rec in enumerate(db):
                with preview_cols[i % 4]:
                    img = pil_open_safe(rec.get("crop_path", ""))
                    if img:
                        st.image(img, use_container_width=True)

                    st.caption(f"{rec.get('zara_category', '?')}")
                    st.caption(f"Color: {rec.get('color', '')}")

                    if rec["item_id"] == st.session_state["selected_base_id"]:
                        st.success("Selected ✅")

                    if st.button("Use this item", key=f"select_{rec['item_id']}"):
                        st.session_state["selected_base_id"] = rec["item_id"]

            base_id = st.session_state.get("selected_base_id")

            selected_item = next((x for x in db if x["item_id"] == base_id), None)
            if selected_item:
                st.markdown("### Selected base item")
                c1, c2 = st.columns([1, 2])
                with c1:
                    img = pil_open_safe(selected_item.get("crop_path", ""))
                    if img:
                        st.image(img, use_container_width=True)
                with c2:
                    st.write(f"**Category:** {selected_item.get('zara_category', '?')}")
                    st.write(f"**Color:** {selected_item.get('color', '')}")
                    st.write(f"**ID:** `{selected_item.get('item_id', '')}`")
            else:
                st.info("Select a garment from the gallery above.")

            if st.button("Complete outfit", type="primary"):
                if not base_id:
                    st.warning("Please select a base garment first.")
                else:
                    results = complete_outfit(base_id, event_key, top_k=topk)
                    if not results:
                        st.warning("No outfits found.")
                    else:
                        for i, r in enumerate(results, start=1):
                            score = r["score"]
                            items = r["items"]

                            st.markdown(
                                f"## Completion #{i} — Score {score.total:.3f} ({'⭐' * score.rating_stars})"
                            )

                            cols = st.columns(len(items))
                            for j, it in enumerate(items):
                                with cols[j]:
                                    img = pil_open_safe(it.get("crop_path", ""))
                                    if img:
                                        st.image(img, use_container_width=True)
                                    st.caption(f"{it['zara_category']} | {it['item_id']}")
                                    st.caption(f"Color: {it.get('color', '')}")

                            with st.expander("Score breakdown & explanation"):
                                st.json({
                                    "total": score.total,
                                    "rating_stars": score.rating_stars,
                                    "category_score": score.category_score,
                                    "event_score": score.event_score,
                                    "formality_score": score.formality_score,
                                    "color_score": score.color_score,
                                    "style_score": score.style_score,
                                    "yolo_consistency_score": score.yolo_consistency_score,
                                    "learned_score": score.learned_score,
                                    "explanation": score.explanation,
                                })

                            st.divider()