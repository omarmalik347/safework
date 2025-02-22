import torch
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Ensure YOLOv5 is installed
try:
    import yolov5
except ImportError:
    st.error("YOLOv5 is not installed. Install it using `pip install git+https://github.com/ultralytics/yolov5.git`")
    st.stop()

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return yolov5.load("keremberke/yolov5m-construction-safety")

model = load_model()

# Set model parameters
model.conf = 0.15  # Confidence threshold
model.iou = 0.30  # IoU threshold
model.agnostic = False
model.multi_label = False
model.max_det = 1000

st.title("Construction Safety Detection")

def upload_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        return Image.open(uploaded_file).convert("RGB")
    return None

def draw_boxes_on_image(image, results):
    image_np = np.array(image)
    
    # Convert image to BGR for OpenCV
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    for _, row in results.iterrows():
        x1, y1, x2, y2, conf, cls = map(int, [row["xmin"], row["ymin"], row["xmax"], row["ymax"], row["confidence"], row["class"]])
        label = f"{model.names[cls]}: {conf:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

img = upload_image()

if img is not None:
    results = model(img)
    results_df = results.pandas().xyxy[0]  # Convert results to Pandas DataFrame

    if not results_df.empty:
        img_with_boxes = draw_boxes_on_image(img, results_df)
        st.image(img_with_boxes, caption="Processed Image with Bounding Boxes", use_container_width=True)
        st.write(f"Detected {len(results_df)} objects")
    else:
        st.write("No objects detected.")
else:
    st.write("Please upload an image to perform detection.")
