#pip install streamlit torch torchvision pillow numpy
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance
import torch
import numpy as np

# ✅ This must be first
st.set_page_config(page_title="🧠 Object Detector")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

# App title
st.title("🔍 Object Detector with Label Buttons")

# Upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Track selected class
if "selected_class" not in st.session_state:
    st.session_state.selected_class = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    results = model(np.array(image))
    detections = results.pandas().xyxy[0]

    if detections.empty:
        st.warning("No objects detected.")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        unique_classes = sorted(set(detections['name']))

        # Layout: Left = buttons, Right = image
        col1, col2 = st.columns([1, 3], gap="large")

        with col1:
            for obj_class in unique_classes:
                if st.button(obj_class.capitalize(), key=f"btn_{obj_class}"):
                    st.session_state.selected_class = obj_class

        with col2:
            selected_class = st.session_state.selected_class
            if selected_class:
                enhancer = ImageEnhance.Brightness(image)
                faded = enhancer.enhance(0.3)
                result_img = faded.copy()
                draw = ImageDraw.Draw(result_img)

                for _, row in detections.iterrows():
                    if row['name'] == selected_class:
                        box = list(map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']]))
                        cropped = image.crop(box).convert("RGB")
                        result_img.paste(cropped, box=tuple(box))
                        draw.rectangle(box, outline='red', width=3)
                        draw.text((box[0], box[1] - 10), row['name'], fill='red')

                st.image(result_img, use_column_width=True)
            else:
                st.image(image, caption="Uploaded Image", use_column_width=True)