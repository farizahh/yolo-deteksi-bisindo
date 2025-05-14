import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import gtts
import tempfile
import os

# Load YOLOv8 model
model = YOLO("bisindo_yolov8.pt")  # Ganti dengan path model yang sesuai

# Set title
st.title("Penerjemah Bahasa Isyarat BISINDO")

# Kamera Input (langsung menangkap gambar)
camera = st.camera_input("Arahkan kamera ke gerakan tangan")

# State untuk label terakhir
if "last_label" not in st.session_state:
    st.session_state.last_label = ""

if camera:
    # Baca frame dari kamera (hasilnya format PIL)
    image = Image.open(camera)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Deteksi objek menggunakan YOLOv8
    results = model.predict(frame)

    # Menyaring hasil deteksi dan menggambar bounding box
    for r in results:
        boxes = r.boxes
        names = r.names

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]

                # Koordinat bounding box
                x1, y1, x2, y2 = box.xyxy[0]  # Koordinat bounding box
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

                # Tampilkan label pada gambar
                frame = cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Tampilkan hasil terjemahan dalam streamlit
                st.markdown(f"### Terjemahan: {label}")

                # Jika label berubah, buat audio baru dan putar
                if label != st.session_state.last_label:
                    tts = gtts.gTTS(label)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                        tts.save(f.name)
                        st.audio(f.name, format="audio/mp3")
                    st.session_state.last_label = label

    # Tampilkan frame dengan bounding box yang sudah digambar
    st.image(frame, channels="BGR")
