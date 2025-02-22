import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# Fix OpenCV headless issue
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

st.title("YOLOv5 Object Detection with Streamlit")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    try:
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        return None

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image for YOLOv5 processing
    img_array = np.array(image)
    results = model(img_array) if model else None

    if results:
        st.subheader("Detection Results")
        results.render()
        st.image(results.ims[0], caption="Detected Objects", use_column_width=True)
