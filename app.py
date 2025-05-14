import streamlit as st

st.title("Penerjemah Bahasa Isyarat BISINDO")

# Contoh hasil prediksi dari YOLOv8 (kamu bisa ganti ini dengan hasil modelmu)
deteksi = "Halo"

# Tampilkan hasil deteksi
st.markdown(f"## Terjemahan: {deteksi}")

# Sisipkan JavaScript untuk bicara otomatis
st.components.v1.html(f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{deteksi}");
        msg.lang = 'id-ID';  // Bahasa Indonesia
        window.speechSynthesis.speak(msg);
    </script>
""", height=0)
