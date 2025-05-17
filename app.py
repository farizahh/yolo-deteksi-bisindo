import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import gtts
import tempfile

# Load model YOLOv8 (ganti path sesuai model kamu)
model = YOLO("bisindo_yolov8.pt")

st.title("Penerjemah Bahasa Isyarat BISINDO")

# --- Sidebar ---
st.sidebar.header("Metode Input")
option = st.sidebar.button("Pilih metode input gambar:", ("Ambil Foto Kamera", "Upload Gambar"))

frame = None  # Variabel untuk menyimpan frame gambar

if option == "Ambil Foto Kamera":
    camera = st.camera_input("Arahkan kamera ke gerakan tangan")
    if camera:
        image = Image.open(camera)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

elif option == "Upload Gambar":
    upload = st.file_uploader("Upload gambar tangan", type=["jpg", "jpeg", "png"])
    if upload:
        image = Image.open(upload)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

if frame is not None:
    # Prediksi menggunakan YOLOv8
    results = model.predict(frame)

    # Variabel untuk menampilkan label terakhir (state agar suara tidak terus menerus)
    if "last_label" not in st.session_state:
        st.session_state.last_label = ""

    # Proses hasil deteksi
    for r in results:
        boxes = r.boxes
        names = r.names

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]

                # Ambil koordinat bounding box dalam integer
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                # Gambar bounding box dan label pada frame
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                frame = cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Tampilkan hasil terjemahan teks
                st.markdown(f"### Terjemahan: {label}")

                # Putar suara hanya jika label berbeda dari sebelumnya
                if label != st.session_state.last_label:
                    tts = gtts.gTTS(label, lang='id')  # 'id' untuk bahasa Indonesia
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                        tts.save(f.name)
                        st.audio(f.name, format="audio/mp3")
                    st.session_state.last_label = label

    # Tampilkan gambar dengan bounding box
    st.image(frame, channels="BGR")
else:
    st.info("Silakan pilih metode input dan berikan gambar tangan untuk diterjemahkan.")
