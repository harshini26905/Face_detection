"""
main.py — Webcam Computer Vision Starter
Press 1: Face Detection
Press 2: Motion Detection
Press 3: Edge Detection
Press 4: Color Detection (tracks green by default)
Press Q: Quit
"""

import cv2
from face_detector import detect_faces
from motion_detector import detect_motion
from edge_detector import detect_edges
from color_detector import detect_color

def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mode = 1  # Start in face detection mode
    prev_frame = None  # Needed for motion detection

    mode_names = {
        1: "Face Detection  [press 2/3/4 to switch]",
        2: "Motion Detection  [press 1/3/4 to switch]",
        3: "Edge Detection  [press 1/2/4 to switch]",
        4: "Color Detection (green)  [press 1/2/3 to switch]",
    }

    print("Webcam CV Project started!")
    print("Press 1-4 to switch modes, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip frame so it feels like a mirror
        frame = cv2.flip(frame, 1)

        # Run the selected detector
        if mode == 1:
            output = detect_faces(frame)
        elif mode == 2:
            output, prev_frame = detect_motion(frame, prev_frame)
        elif mode == 3:
            output = detect_edges(frame)
        elif mode == 4:
            output = detect_color(frame)
        else:
            output = frame

        # Show mode label on screen
        label = mode_names.get(mode, "")
        cv2.putText(output, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Webcam CV Project", output)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("Quitting...")
            break
        elif key == ord('1'):
            mode = 1
            print("Switched to: Face Detection")
        elif key == ord('2'):
            mode = 2
            prev_frame = None  # Reset motion baseline
            print("Switched to: Motion Detection")
        elif key == ord('3'):
            mode = 3
            print("Switched to: Edge Detection")
        elif key == ord('4'):
            mode = 4
            print("Switched to: Color Detection")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
