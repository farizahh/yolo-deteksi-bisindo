import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

# Load model
model = YOLO("bisindo_yolov8.pt")

# Judul halaman
st.title("Penerjemah Bahasa Isyarat BISINDO")

# Kamera input
camera = st.camera_input("Arahkan kamera ke gerakan tangan")

# Simpan label terakhir
if "last_label" not in st.session_state:
    st.session_state.last_label = ""

# Jika ada input kamera
if camera:
    image = Image.open(camera)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Deteksi pakai YOLO
    results = model.predict(frame)

    for r in results:
        boxes = r.boxes
        names = r.names

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]

                # Tampilkan hasil di layar
                st.markdown(f"### Terjemahan: {label}")

                # Jika label berubah, trigger suara
                if label != st.session_state.last_label:
                    # Sematkan JavaScript untuk bicara
                    js_code = f"""
                        <script>
                        var msg = new SpeechSynthesisUtterance("{label}");
                        window.speechSynthesis.speak(msg);
                        </script>
                    """
                    st.components.v1.html(js_code)
                    st.session_state.last_label = label
