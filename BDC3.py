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
model = YOLO(r"model_yolo.pt") 

# Fungsi untuk Halaman 1
def halaman_penjelasan():
    # Tampilkan judul di tengah halaman
    st.markdown("""
    <div style="text-align: center;">
        <h1>Penjelasan Kelas Gambar</h1>
    </div>
    """, unsafe_allow_html=True)

    # Path ke gambar
    image_paths = [
        r"Image/None.jpg",
        r"Image/Fire.jpg",
        r"Image/Smoke.jpg",
        r"Image/Smoke and Fire.jpg"
    ]
    
    descriptions = [
        """Kelas NONE. Pengamatan visual yang cermat terhadap keseluruhan gambar tidak
        memberikan indikasi adanya tanda-tanda kebakaran yang biasanya 
        tampak dalam situasi darurat, seperti jejak hangus pada permukaan,
        bekas-bekas material yang terbakar, atau adanya asap tipis yang
        membubung dari area tertentu. Selain itu, tidak ada tanda-tanda
        kehadiran sumber panas yang tidak biasa, seperti kilatan cahaya atau
        perubahan warna yang dapat menandakan suhu tinggi. Seluruh elemen yang
        terlihat dalam gambar tampak normal dan tidak menunjukkan aktivitas yang
        berpotensi berbahaya. Oleh karena itu, dapat disimpulkan dengan tingkat 
        keyakinan yang tinggi bahwa kondisi di lokasi tersebut saat ini sepenuhnya
        aman dan tidak menunjukkan adanya ancaman kebakaran yang perlu diwaspadai.""",

        """Kelas Fire. Gambar ini menunjukkan keberadaan api yang mendominasi area tersebut, 
        dengan nyala api yang jelas terlihat serta indikasi bahwa api tersebut dapat
        berkembang dengan cepat jika tidak segera dikendalikan. Warna-warna cerah 
        seperti merah dan oranye mencolok mengindikasikan bahwa api berada dalam fase 
        aktif, dengan kemungkinan bahaya besar terhadap lingkungan sekitarnya jika tidak
        diatasi dengan segera.""",

        """Kelas Smoke. Gambar ini secara jelas menunjukkan dominasi adanya asap yang tebal dan membubung
        ke udara, menciptakan suasana yang misterius dan mendalam. Asap tersebut tampak 
        menyelimuti area sekitarnya, dengan warna abu-abu gelap yang menunjukkan potensi adanya 
        kebakaran atau sumber panas lainnya di dekatnya. Kehadiran asap ini bisa menjadi indikasi 
        bahwa suatu proses pembakaran sedang berlangsung, baik itu berupa kebakaran kecil, 
        pembakaran sampah, atau mungkin bahkan aktivitas industri. Dengan demikian, gambaran 
        ini mengundang perhatian dan menimbulkan pertanyaan tentang asal-usul asap tersebut 
        dan potensi bahayanya terhadap lingkungan sekitar.""",

        """Kelas Smoke and Fire. Gambar ini secara mencolok menunjukkan keberadaan api dan asap yang muncul secara 
        bersamaan, menciptakan pemandangan yang dramatis dan penuh ketegangan. Nyala api yang 
        berkobar dengan warna merah dan oranye yang mencolok tampak menari-nari di antara kepulan 
        asap yang tebal dan gelap, yang membubung tinggi ke langit. Kombinasi antara api yang aktif 
        dan asap yang menyelimuti area tersebut memberikan indikasi bahwa suatu proses pembakaran 
        yang signifikan sedang berlangsung. Selain itu, asap yang terlihat dapat mengisyaratkan 
        potensi bahaya yang lebih besar, karena dapat menandakan adanya material yang terbakar atau 
        zat berbahaya di sekitarnya. Pemandangan ini tidak hanya menarik perhatian, tetapi juga 
        menimbulkan kekhawatiran mengenai keselamatan dan potensi ancaman yang ditimbulkan oleh 
        kebakaran yang mungkin terjadi di lokasi tersebut"""
    ]
    
    # Tampilkan gambar dan penjelasan menggunakan layout kolom
    for i in range(4):  # Loop untuk setiap gambar
        image = Image.open(image_paths[i])

        # Buat dua kolom: kolom kiri untuk gambar, kolom kanan untuk penjelasan
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
            # Konversi gambar ke format numpy array
            img_array = np.array(image)

            # Lakukan deteksi menggunakan model YOLOv8
            results = model(img_array)

            # Ambil nama kelas dan probabilitas
            names_dict = results[0].names
            probs = results[0].probs.data.tolist()
            class_name = names_dict[np.argmax(probs)]
            confidence = np.max(probs)

            st.markdown(f"""
            <div style="text-align:center; font-size:30px; font-weight:bold; color:#FF5733;">
                Kelas : {class_name}
            </div>
            """, unsafe_allow_html=True)

            st.image(results[0].plot(), caption='Hasil Deteksi', use_container_width=True)  # Ganti ke use_container_width

# Menampilkan halaman berdasarkan pilihan
if selected == "Tentang":
    halaman_penjelasan()
elif selected == "Klasifikasi Gambar":
    halaman_klasifikasi()
