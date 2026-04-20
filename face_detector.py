"""
face_detector.py — Detects faces using OpenCV's built-in Haar Cascade classifier.
No extra downloads needed — the XML files come with OpenCV.
"""

import cv2

# Load the pre-trained face detector (built into OpenCV)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Optional: also detect eyes inside each face
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


def detect_faces(frame):
    """
    Takes a BGR frame, returns the same frame with faces (and eyes) boxed.
    """
    output = frame.copy()

    # Convert to grayscale — Haar cascades work on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # scaleFactor: how much to shrink image each pass (1.1 = 10% smaller)
    # minNeighbors: how many detections needed before confirming a face (higher = stricter)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # Draw blue rectangle around face
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 100, 0), 2)

        # Label above the face
        cv2.putText(output, "Face", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        # Look for eyes only inside the face region (faster & more accurate)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = output[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20)
        )
        for (ex, ey, ew, eh) in eyes:
            center = (ex + ew // 2, ey + eh // 2)
            radius = ew // 2
            cv2.circle(roi_color, center, radius, (0, 200, 255), 2)

    # Show count
    count_text = f"Faces found: {len(faces)}"
    cv2.putText(output, count_text, (10, output.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output
