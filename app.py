import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import gtts
import tempfile

model = YOLO("bisindo_yolov8.pt")

st.title("Penerjemah Bahasa Isyarat BISINDO")

# Kamera input di sidebar
camera = st.sidebar.camera_input("Ambil foto dengan kamera")

# Upload gambar di sidebar
upload = st.sidebar.file_uploader("Atau upload gambar", type=["jpg", "jpeg", "png"])

frame = None

if camera:
    image = Image.open(camera)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
elif upload:
    image = Image.open(upload)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

if frame is not None:
    results = model.predict(frame)

    if "last_label" not in st.session_state:
        st.session_state.last_label = ""

    for r in results:
        boxes = r.boxes
        names = r.names

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                frame = cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                st.markdown(f"### Terjemahan: {label}")

                if label != st.session_state.last_label:
                    tts = gtts.gTTS(label, lang='id')
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                        tts.save(f.name)
                        st.audio(f.name, format="audio/mp3")
                    st.session_state.last_label = label

    st.image(frame, channels="BGR")
else:
    st.info("Silakan ambil foto dengan kamera atau upload gambar untuk diterjemahkan.")
