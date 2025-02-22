import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO  # Import YOLO from ultralytics package

# Set OpenCV headless mode
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

st.title("YOLOv5 Object Detection with Streamlit")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov5s.pt")  # Load YOLOv5 model
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        return None

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Convert image for YOLOv5 processing
    img_array = np.array(image)
    
    if model:
        results = model(img_array)
        st.subheader("Detection Results")

        for result in results:
            result.save(filename="output.jpg")  # Save detected image
            st.image("output.jpg", caption="Detected Objects", use_column_width=True)
