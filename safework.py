import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO  # Updated import for Ultralytics YOLO

# Load YOLOv5 Model
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    return YOLO("yolov5m.pt")  # Pre-trained YOLOv5 model

model = load_model()

# Function to upload an image
def upload_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return image
    return None

# Function to draw bounding boxes on image
def draw_boxes(image, results):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]}: {conf:.2f}"

            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Streamlit UI
st.title("Construction Safety Detection")

img = upload_image()

if img is not None:
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run detection
    results = model.predict(img, imgsz=960, conf=0.25, iou=0.30)

    # Draw bounding boxes
    img_with_boxes = draw_boxes(img, results)

    st.image(img_with_boxes, caption="Processed Image with Bounding Boxes", use_column_width=True)

    # Display detected objects
    detected_classes = [model.names[int(box.cls[0])] for r in results for box in r.boxes]
    if detected_classes:
        st.write(f"### Detected Objects:")
        st.write(", ".join(set(detected_classes)))
    else:
        st.write("No objects detected.")
else:
    st.write("Please upload an image to perform detection.")
