import cv2
import base64
import numpy as np
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Yor_API_Key_Here"
)


VIDEO_PATH = r"C:\Users\DELL\Desktop\fine_tuning_skuld_operation\skuld_test1.mp4"
OUTPUT_PATH = r"C:\Users\DELL\Desktop\fine_tuning_skuld_operation\skuld_operation_output_video105.mp4"
MODEL_ID = "yolo11-moulc/5"

# Pre-defined distinct color palette
color_palette = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
]


class_colors = {}
def get_color_for_class(class_name):
    if class_name not in class_colors:
        if len(class_colors) < len(color_palette):
            class_colors[class_name] = color_palette[len(class_colors)]
        else:
            class_colors[class_name] = tuple(np.random.choice(range(256), size=3))
    return class_colors[class_name]


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            frame_count += 1

            _, buffer = cv2.imencode(".jpg", frame)
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            try:
                result = CLIENT.infer(img_base64, model_id=MODEL_ID)
            except Exception as e:
                print(f"Inference error: {e}")
                out.write(frame)
                continue

            predictions = result.get("predictions", [])
            for pred in predictions:
                x = int(pred["x"] - pred["width"] / 2)
                y = int(pred["y"] - pred["height"] / 2)
                width = int(pred["width"])
                height = int(pred["height"])
                class_name = pred["class"]
                confidence = pred["confidence"]
                color = get_color_for_class(class_name)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Processing Video", frame)

            
            out.write(frame)

            
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"Processed video saved to {OUTPUT_PATH}")
