import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load model YOLOv8
model = YOLO("bisindo_yolov8.pt")  # Pastikan model tersedia di direktori yang benar

# Title of the Streamlit app
st.title("Penerjemah Bahasa Isyarat BISINDO")

# Camera Input Streamlit
camera = st.camera_input("Arahkan kamera ke gerakan tangan")

# Check if camera is available
if camera is not None:
    # Baca frame dari kamera (PIL image)
    image = Image.open(camera)
    frame = np.array(image)
    
    # Deteksi menggunakan model YOLOv8
    results = model.predict(frame)
    
    # Loop through results and draw bounding boxes
    for result in results:
        boxes = result.boxes
        names = result.names
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                
                # Gambar bounding box pada frame
                frame = result.plot()
                
                # Tampilkan label
                st.markdown(f"### Terjemahan: {label}")
    
    # Convert frame to RGB and display it in Streamlit
    st.image(frame, channels="BGR", caption="Detected Sign Language", use_column_width=True)
else:
    st.warning("Tidak ada input dari kamera. Pastikan kamera terhubung dengan benar.")
