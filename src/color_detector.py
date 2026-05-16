"""
Dominant color detection for garment crops.

"""

from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np
from sklearn.cluster import KMeans


# HSV ranges (OpenCV HSV: H 0-179, S 0-255, V 0-255)
COLOR_RANGES = {
    "black":  ((0, 0, 0),    (179, 255, 50)),
    "white":  ((0, 0, 210),  (179, 40, 255)),
    "grey":   ((0, 0, 51),   (179, 40, 209)),

    "red1":   ((0, 70, 50),   (8, 255, 255)),
    "red2":   ((170, 70, 50), (179, 255, 255)),

    "orange": ((9, 70, 70),   (20, 255, 255)),
    "yellow": ((21, 70, 70),  (34, 255, 255)),
    "green":  ((35, 50, 50),  (84, 255, 255)),
    "blue":   ((85, 50, 50),  (124, 255, 255)),
    "purple": ((125, 50, 50), (149, 255, 255)),
    "pink":   ((150, 50, 80), (169, 255, 255)),

    "brown":  ((10, 80, 30),  (20, 255, 170)),
}


def _match_color(hsv_pixel: Tuple[int, int, int]) -> str:
    h, s, v = hsv_pixel

    # neutral colors first
    if s < 40:
        if v >= 210:
            return "white"
        elif v >= 51:
            return "grey"
        else:
            return "black"

    # brown first because brown is like a dark orange
    if 10 <= h <= 20 and s >= 80 and 30 <= v <= 170:
        return "brown"

    # red wrap
    if (0 <= h <= 8 or 170 <= h <= 179) and s >= 70 and v >= 50:
        return "red"

    if 9 <= h <= 20 and s >= 70 and v >= 70:
        return "orange"

    if 21 <= h <= 34 and s >= 70 and v >= 70:
        return "yellow"

    if 35 <= h <= 84 and s >= 50 and v >= 50:
        return "green"

    if 85 <= h <= 124 and s >= 50 and v >= 50:
        return "blue"

    if 125 <= h <= 149 and s >= 50 and v >= 50:
        return "purple"

    if 150 <= h <= 169 and s >= 50 and v >= 80:
        return "pink"

    return "unknown"


def _filter_pixels(hsv_pixels: np.ndarray) -> np.ndarray:
    """
    If most pixels are low-saturation => likely white/grey garment => keep all
    Else remove low-saturation background + extreme highlights + deep shadows
    """
    s = hsv_pixels[:, 1]
    v = hsv_pixels[:, 2]

    low_sat_ratio = float(np.mean(s < 40))

    if low_sat_ratio > 0.60:
        return hsv_pixels

    mask = (
        (s > 40) &
        (v > 30) &
        (v < 245)
    )

    filtered = hsv_pixels[mask]
    if filtered.shape[0] < 500:
        return hsv_pixels
    return filtered


def detect_dominant_color(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        return "unknown"

    img = cv2.resize(img, (220, 220), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    pixels = hsv.reshape((-1, 3))
    pixels = _filter_pixels(pixels)

    k = 3 if pixels.shape[0] >= 3000 else 2
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(pixels)

    counts = np.bincount(km.labels_)
    dominant = km.cluster_centers_[int(np.argmax(counts))]
    dominant_hsv = (int(dominant[0]), int(dominant[1]), int(dominant[2]))

    print("Dominant HSV:", dominant_hsv)  # for testing

    return _match_color(dominant_hsv)