import streamlit as st
import streamlit_option_menu
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Gambar",
    page_icon=":🌀:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Buat menu utama di sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Menu Utama",
        options=["Tentang", "Klasifikasi Gambar"],
        icons=["house", "book"],
        menu_icon="cast",
        default_index=0,
    )

# Muat model YOLO
# Gantilah path dengan path relatif di folder aplikasi Streamlit
model = YOLO('model_yolo.pt')  # Pastikan pathnya sesuai

# Fungsi untuk Halaman 1
def halaman_penjelasan():
    st.markdown("""
    <div style="text-align: center;">
        <h1>Penjelasan Kelas Gambar</h1>
    </div>
    """, unsafe_allow_html=True)

    # Path ke gambar
    image_paths = [
        "./Image/None.jpg",
        "./Image/Fire.jpg",
        "./Image/Smoke.jpg",
        "./Image/Smoke and Fire.jpg"
    ]
    
    descriptions = [
        """Kelas NONE. Pengamatan visual yang cermat terhadap keseluruhan gambar tidak memberikan indikasi adanya tanda-tanda kebakaran yang biasanya ...""",
        """Kelas Fire. Gambar ini menunjukkan keberadaan api yang mendominasi area tersebut ...""",
        """Kelas Smoke. Gambar ini secara jelas menunjukkan dominasi adanya asap yang tebal ...""",
        """Kelas Smoke and Fire. Gambar ini secara mencolok menunjukkan keberadaan api dan asap yang muncul secara bersamaan ..."""
    ]
    
    # Tampilkan gambar dan penjelasan menggunakan layout kolom
    for i in range(4):  # Loop untuk setiap gambar
        image = Image.open(image_paths[i])

        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, use_container_width=True)  # Ganti ke use_container_width
        
        with col2:
            st.markdown(f"""
            <div style="text-align: justify; padding-top: 5px; padding-right: 10px;">
                {descriptions[i]}
            </div>
            """, unsafe_allow_html=True)

# Fungsi untuk Halaman 2
def halaman_klasifikasi():
    st.markdown("""
    <div style="text-align: center;">
        <h1>Klasifikasi Gambar</h1>
    </div>
    """, unsafe_allow_html=True)
    st.write("Di sini pengguna dapat mengunggah gambar untuk diklasifikasi.")
    
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah.', use_container_width=True)  # Ganti ke use_container_width

        if st.button('Deteksi'):
            img_array = np.array(image)

            # Lakukan deteksi menggunakan model YOLOv8
            results = model(img_array)

            names_dict = results[0].names
            probs = results[0].probs.data.tolist()
            class_name = names_dict[np.argmax(probs)]
            confidence = np.max(probs)

            st.markdown(f"""
            <div style="text-align:center; font-size:30px; font-weight:bold; color:#FF5733;">
                Kelas : {class_name}
            </div>
            """, unsafe_allow_html=True)

            st.image(results[0].plot(), caption='Hasil Deteksi', use_container_width=True)

# Menampilkan halaman berdasarkan pilihan
if selected == "Tentang":
    halaman_penjelasan()
elif selected == "Klasifikasi Gambar":
    halaman_klasifikasi()
