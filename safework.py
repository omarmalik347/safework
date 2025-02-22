import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO  

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

st.title("Safe Work Detection Model")


@st.cache_resource
def load_model():
    try:
        model = YOLO("yolo11s.pt")  # Load model
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image)

    if model:
        results = model(img_array, conf=0.15) 
        st.subheader("Detection Results")

        detected_objects = []
        for result in results:
            result.save(filename="output.jpg")  # Save detected image
            st.image("output.jpg", caption="Detected Objects")

            # Extract detected objects with confidence scores
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = box
                if confidence >= 0.15:  # Ensure confidence threshold is met
                    detected_objects.append((model.names[int(class_id)], confidence))

        if detected_objects:
            st.write("### Detected Objects:")
            for obj, conf in detected_objects:
                st.write(f"🔹 {obj} - {conf:.2f} confidence")
        else:
            st.write("⚠️ No objects detected with confidence ≥ 0.15.")
