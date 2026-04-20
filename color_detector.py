"""
color_detector.py — Tracks a specific color (green by default) using HSV color space.
HSV (Hue, Saturation, Value) is much better than BGR for color detection.
Change the HSV ranges below to track any color you like!
"""

import cv2
import numpy as np


# ── Color range presets ───────────────────────────────────────────────────────
# OpenCV HSV: Hue 0-179, Saturation 0-255, Value 0-255
COLOR_RANGES = {
    "green":  {"lower": (35, 60, 60),  "upper": (85, 255, 255),  "bgr": (0, 200, 0)},
    "red":    {"lower": (0,  120, 70),  "upper": (10, 255, 255),  "bgr": (0, 0, 220)},
    "blue":   {"lower": (100, 80, 50),  "upper": (130, 255, 255), "bgr": (220, 80, 0)},
    "yellow": {"lower": (20, 100, 100), "upper": (35, 255, 255),  "bgr": (0, 220, 220)},
    "orange": {"lower": (10, 100, 100), "upper": (20, 255, 255),  "bgr": (0, 140, 255)},
}

# Change this to "red", "blue", "yellow", or "orange" to track a different color
ACTIVE_COLOR = "green"


def detect_color(frame, color_name=ACTIVE_COLOR):
    """
    Finds and outlines the largest blob of the target color in the frame.
    Returns the frame with an overlay and bounding box.
    """
    cfg = COLOR_RANGES.get(color_name, COLOR_RANGES["green"])
    output = frame.copy()

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array(cfg["lower"])
    upper = np.array(cfg["upper"])

    # Create a binary mask: white where color matches, black elsewhere
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up the mask: remove small noise blobs
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)  # fill gaps

    # Find all blobs that match the color
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Focus on the largest matching blob
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 800:  # Ignore tiny specks
            (x, y, w, h) = cv2.boundingRect(largest)
            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), cfg["bgr"], 2)

            # Draw centroid dot
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(output, (cx, cy), 6, cfg["bgr"], -1)

            # Label
            label = f"{color_name.capitalize()} object  area={int(area)}px"
            cv2.putText(output, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cfg["bgr"], 2)

            # Show a small mask preview in top-right corner
            h_frame, w_frame = frame.shape[:2]
            preview_w, preview_h = 160, 90
            mask_preview = cv2.resize(mask, (preview_w, preview_h))
            mask_preview_bgr = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR)
            px = w_frame - preview_w - 5
            py = 40
            output[py:py + preview_h, px:px + preview_w] = mask_preview_bgr
            cv2.putText(output, "mask", (px + 4, py + preview_h - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Footer info
    tip = f"Tracking: {color_name}  |  Edit ACTIVE_COLOR in color_detector.py to change"
    cv2.putText(output, tip, (10, output.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    return output
