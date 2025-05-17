import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import gtts
import tempfile

# Load YOLOv8 model
model = YOLO("bisindo_yolov8.pt")  # Ganti dengan path modelmu

# Judul halaman
st.title("Penerjemah Bahasa Isyarat BISINDO")

# Sidebar: pilih metode input
mode = st.sidebar.selectbox(
    'Pilih metode input gambar:',
    ('Ambil Foto Kamera', 'Upload Gambar')
)

# Inisialisasi variabel frame
frame = None

# Ambil gambar berdasarkan mode yang dipilih
if mode == 'Ambil Foto Kamera':
    camera = st.sidebar.camera_input("Ambil foto dari kamera")
    if camera:
        image = Image.open(camera)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

elif mode == 'Upload Gambar':
    upload = st.sidebar.file_uploader("Upload gambar tangan", type=["jpg", "jpeg", "png"])
    if upload:
        image = Image.open(upload)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Jika ada gambar, jalankan deteksi
if frame is not None:
    results = model.predict(frame)

    # Simpan label sebelumnya agar suara tidak diputar berulang
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

                # Tampilkan hasil deteksi
                st.markdown(f"### Terjemahan: {label}")

                # Putar suara jika label berubah
                if label != st.session_state.last_label:
                    tts = gtts.gTTS(label, lang='id')
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                        tts.save(f.name)
                        st.audio(f.name, format="audio/mp3")
                    st.session_state.last_label = label

    # Tampilkan gambar dengan bounding box
    st.image(frame, channels="BGR")
else:
    st.info("Silakan ambil gambar atau upload untuk mendeteksi gerakan tangan.")
