import os
import cv2
from inference_sdk import InferenceHTTPClient
import numpy as np

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Your API Key Here" 
)

# Image path and model ID
IMAGE_PATH = r"C:\Users\DELL\Desktop\fine_tuning_skuld_operation\OIP.jpg"
MODEL_ID = "yolo11-moulc/5"

try:
    # Perform inference
    result = CLIENT.infer(IMAGE_PATH, model_id=MODEL_ID)

    # Load the image
    image = cv2.imread(IMAGE_PATH)

    # Pre-defined distinct color palette
    color_palette = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Navy
    ]

    # Create a color map for classes
    class_colors = {}
    def get_color_for_class(class_name):
        if class_name not in class_colors:
            # Assign a color from the palette or generate a random unique color
            if len(class_colors) < len(color_palette):
                class_colors[class_name] = color_palette[len(class_colors)]
            else:
                # Generate a unique random color if the palette is exhausted
                class_colors[class_name] = tuple(np.random.choice(range(256), size=3))
        return class_colors[class_name]

    # Extract predictions
    predictions = result.get("predictions", [])
    for pred in predictions:
        # Get bounding box details
        x = int(pred["x"] - pred["width"] / 2)
        y = int(pred["y"] - pred["height"] / 2)
        width = int(pred["width"])
        height = int(pred["height"])
        class_name = pred["class"]
        confidence = pred["confidence"]

        # Get color for the class
        color = get_color_for_class(class_name)

        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
        label = f"{class_name} ({confidence:.2f})"
        
        # Add the label above the bounding box
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image
    cv2.imshow("Image with Bounding Boxes", image)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")
