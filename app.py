import cv2
from ultralytics import YOLO
from gtts import gTTS
import streamlit as st
import tempfile
import os
from PIL import Image
import numpy as np

# Load YOLOv8 model
model = YOLO("bisindo_yolov8.pt")

# Title
st.title("Penerjemah Bahasa Isyarat BISINDO")

# Kamera
camera = st.camera_input("Arahkan kamera ke gerakan tangan")

# State untuk label terakhir
if "last_label" not in st.session_state:
    st.session_state.last_label = ""

if camera:
    # Baca frame dari kamera (hasilnya format PIL)
    image = Image.open(camera)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Deteksi
    results = model.predict(frame)
    for r in results:
        boxes = r.boxes
        names = r.names
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]

                # Tampilkan label
                st.markdown(f"### Terjemahan: {label}")

                # Jika label berubah, buat audio baru
                if label != st.session_state.last_label:
                    tts = gTTS(label)
                    tts.save("output.mp3")
                    st.audio("output.mp3", format="audio/mp3")
                    st.session_state.last_label = label
