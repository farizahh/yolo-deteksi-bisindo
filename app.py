import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import gtts
import tempfile

# Load model YOLOv8 (ganti path sesuai model kamu)
model = YOLO("bisindo_yolov8.pt")

# Judul halaman utama
st.title("Penerjemah Bahasa Isyarat BISINDO")

# --- Sidebar ---
st.sidebar.header("Metode Input")
option = st.sidebar.radio("Pilih metode input gambar:", ("Ambil Foto Kamera", "Upload Gambar"))

frame = None  # Variabel untuk menyimpan frame gambar

if option == "Ambil Foto Kamera":
    camera = st.sidebar.camera_input("Arahkan kamera ke gerakan tangan")
    if camera:
        image = Image.open(camera)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

elif option == "Upload Gambar":
    upload = st.sidebar.file_uploader("Upload gambar tangan", type=["jpg", "jpeg", "png"])
    if upload:
        image = Image.open(upload)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# --- Proses deteksi ---
if frame is not None:
    results = model.predict(frame)

    if
