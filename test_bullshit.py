from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\DELL\Desktop\fine_tuning_skuld_operation\runs\detect\train6\weights\best.pt")
image_path = r"C:\Users\DELL\Desktop\fine_tuning_skuld_operation\S4.jpg"
results = model(image_path)

result = results[0]
if result.boxes:
    for box in result.boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}, Bounding Box: {box.xywh}")

result.show()
result.save("output.jpg")