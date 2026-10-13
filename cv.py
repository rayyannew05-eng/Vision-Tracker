
import cv2
import time
from collections import Counter
from ultralytics import YOLO

# ==========================================
# VISION TRACKER — REAL-TIME AI
# ==========================================

print("Loading AI model... First run may download model weights.")

model = YOLO("yolo11n.pt")  # Lightweight pretrained YOLO model

# Change 0 to 1 if your camera is connected as camera 1
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    print("Check your camera connection and close other apps using it.")
    raise SystemExit(1)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

previous_time = time.time()
tracked_ids = set()

print("Vision Tracker started!")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a camera frame.")
            break

        # Run AI detection and track objects across frames
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=416,
            conf=0.35,
            device="cpu",
            verbose=False
        )

        result = results[0]
        output = result.plot()

        # Count objects visible in the current frame
        visible_counts = Counter()

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                visible_counts[class_name] += 1

            # Keep track of IDs observed during this session
            if result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                tracked_ids.update(ids)

        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - previous_time
        fps = 1 / elapsed if elapsed > 0 else 0
        previous_time = current_time

        # Draw a clean information panel
        cv2.rectangle(output, (10, 10), (340, 105), (25, 25, 25), -1)

        cv2.putText(
            output, "VISION TRACKER",
            (22, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.putText(
            output, f"Visible objects: {sum(visible_counts.values())}",
            (22, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1
        )

        cv2.putText(
            output, f"Unique track IDs: {len(tracked_ids)}",
            (22, 82), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1
        )

        cv2.putText(
            output, f"FPS: {fps:.1f}",
            (22, 100), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 255, 0), 1
        )

        # Display detected object names
        if visible_counts:
            summary = " | ".join(
                f"{name}: {count}"
                for name, count in visible_counts.most_common(4)
            )
            cv2.putText(
                output, summary[:75],
                (10, output.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1
            )

        cv2.imshow("Vision Tracker | Press Q to Quit", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Vision Tracker stopped.")

    