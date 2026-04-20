"""
edge_detector.py — Highlights edges in the frame using the Canny algorithm.
Great for understanding image structure and shapes.
"""

import cv2
import numpy as np


def detect_edges(frame):
    """
    Applies Canny edge detection and returns a colourised edge overlay.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur slightly to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    # threshold1 (lower): pixels below this are NOT edges
    # threshold2 (upper): pixels above this ARE definitely edges
    # Pixels in between are edges only if connected to a definite edge
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Convert single-channel edge map to 3-channel so we can colourize it
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Tint edges cyan for a nice look
    # Mask: where edges exist (non-zero), tint those pixels
    mask = edges > 0
    edges_colored[mask] = [255, 220, 0]  # Cyan-yellow tint

    # Blend with original frame for context (50% edge overlay, 50% original)
    output = cv2.addWeighted(frame, 0.4, edges_colored, 0.9, 0)

    # Show edge pixel count as feedback
    edge_count = int(np.sum(mask))
    cv2.putText(output, f"Edge pixels: {edge_count}", (10, output.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output
