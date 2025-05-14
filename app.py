import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
from gtts import gTTS
import tempfile
import os

# Load YOLOv8 model
model = YOLO("bisindo_yolov8.pt")  # Pastikan path benar

# Streamlit title
st.title("📷 Penerjemah Bahasa Isyarat BISINDO")

# Inisialisasi session_state
if "last_label" not in st.session_state:
    st.session_state.last_label = ""

# Ambil input kamera
camera = st.camera_input("📸 Arahkan kamera ke huruf BISINDO")

if camera:
    # Konversi input kamera ke array BGR (OpenCV)
    img_pil = Image.open(camera)
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Prediksi dengan YOLOv8
    results = model.predict(frame)

    for r in results:
        boxes = r.boxes
        names = r.names  # dict class_id -> label

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                conf = float(box.conf[0])  # confidence

                # Tampilkan hanya jika confidence cukup tinggi
                if conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    frame = cv2.putText(
                        frame,
                        f"{label} ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                    st.image(frame, channels="BGR", caption="📍 Deteksi huruf BISINDO")

                    # Jika label baru, putar audio
                    if label != st.session_state.last_label:
                        tts = gTTS(text=label, lang='id')
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                            tts.save(f.name)
                            st.audio(f.name, format="audio/mp3")
                        st.session_state.last_label = label

                    st.success(f"✅ Terjemahan: **{label}**")
                    break  # Ambil satu huruf saja agar tidak tumpang tindih
