"""
motion_detector.py — Detects motion by comparing the current frame to the previous one.
Simple & effective — no ML needed.
"""

import cv2
import numpy as np


def detect_motion(frame, prev_frame):
    """
    Compares current frame to prev_frame to highlight moving regions.
    Returns (output_frame, updated_prev_frame).
    Always pass the returned prev_frame back in the next call.
    """
    output = frame.copy()

    # Convert current frame to grayscale and blur it (reduces noise)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # On the very first call, we have no previous frame yet
    if prev_frame is None:
        return output, gray

    # Compute absolute difference between current and previous frame
    diff = cv2.absdiff(prev_frame, gray)

    # Threshold: pixels that changed more than 25 become white, rest black
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Dilate the white regions to fill small gaps
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours (outlines) of the moving regions
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False
    for contour in contours:
        # Ignore tiny movements (dust, noise)
        if cv2.contourArea(contour) < 1500:
            continue

        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 200, 50), 2)

    # Status text
    status = "MOTION DETECTED!" if motion_detected else "No motion"
    color = (0, 50, 255) if motion_detected else (200, 200, 200)
    cv2.putText(output, status, (10, output.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return output, gray  # Return updated prev_frame
