"""
Sentiment Classification for Telaga Sarangan Tourism
Using Random Forest — Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import joblib
import re
import string
import os
import numpy as np
from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ── Configuration ────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
LABEL_MAP = {0: "Negatif", 1: "Netral", 2: "Positif"}

SLANG_MAP = {
    'gak':'tidak','ga':'tidak','nggak':'tidak','gk':'tidak',
    'ngga':'tidak','tdk':'tidak','g':'tidak','gpp':'tidak apa',
    'emg':'memang','emang':'memang','bgt':'banget','bngt':'banget',
    'dr':'dari','utk':'untuk','tp':'tapi','yg':'yang',
    'sy':'saya','sm':'sama','bs':'bisa','jg':'juga',
    'dg':'dengan','dgn':'dengan','blm':'belum','krn':'karena',
    'sdh':'sudah','udh':'sudah','udah':'sudah','lg':'lagi',
    'hrs':'harus','gitu':'begitu','gt':'begitu','aja':'saja',
    'aj':'saja','bgs':'bagus','bnr':'benar','bnyk':'banyak',
    'org':'orang','jgn':'jangan','jng':'jangan','klo':'kalau',
    'kl':'kalau','kalo':'kalau','trs':'terus','trus':'terus',
    'skrg':'sekarang','skr':'sekarang','bkn':'bukan',
    'tgl':'tanggal','msh':'masih','sgt':'sangat','byk':'banyak',
    'dpn':'depan','blkg':'belakang','brg':'barang',
    'mantep':'mantap','mantab':'mantap','mantabs':'mantap',
    'recomended':'recommended','worthit':'worth it',
    'kyk':'kayak','kayak':'seperti','kaya':'seperti',
    'mantapss':'mantap','mntap':'mantap','mantaps':'mantap',
}


# ── Custom sklearn classes (must match training) ─────────────────────
class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract statistical text features from raw review text."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = pd.DataFrame()
        features['char_count'] = X.apply(len)
        features['word_count'] = X.apply(lambda x: len(str(x).split()))
        features['exclamation_count'] = X.apply(lambda x: x.count('!'))
        features['question_count'] = X.apply(lambda x: x.count('?'))
        features['caps_count'] = X.apply(
            lambda x: sum(1 for c in str(x) if c.isupper()))
        features['avg_word_len'] = X.apply(
            lambda x: np.mean([len(w) for w in str(x).split()])
            if len(str(x).split()) > 0 else 0)
        features['unique_word_count'] = X.apply(
            lambda x: len(set(str(x).lower().split())))
        features['caps_ratio'] = X.apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
        features['digit_count'] = X.apply(
            lambda x: sum(1 for c in str(x) if c.isdigit()))
        features['punctuation_count'] = X.apply(
            lambda x: sum(1 for c in str(x) if c in string.punctuation))
        return features

class RatioStrategy:
    """Fungsi custom untuk mengatur porsi penciptaan data sintetis pada SMOTE."""
    def __init__(self, ratio):
        self.ratio = ratio
        
    def __call__(self, y):
        counts = Counter(y)
        maj_count = max(counts.values())
        target = int(np.floor(maj_count * self.ratio))
        return {k: max(v, target) for k, v in counts.items()}
        
    def __repr__(self):
        return f"RatioStrategy({self.ratio})"



# ── NLP tools ────────────────────────────────────────────────────────
@st.cache_resource
def _load_nlp():
    stemmer = StemmerFactory().create_stemmer()
    sw = set(StopWordRemoverFactory().get_stop_words())
    return stemmer, sw

def preprocess_text(text: str) -> str:
    stemmer, stopwords = _load_nlp()
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'#[\w]+', '', text)
    text = re.sub(r'\d+', '', text)
    p = string.punctuation
    text = text.translate(str.maketrans('', '', p))
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Normalisasi kata (slang)
    words = [SLANG_MAP.get(w, w) for w in text.split()]
    
    # Pisahkan kembali jika ada slang map yang menghasilkan dua kata (misal 'tidak apa')
    normalized_words = []
    for w in words:
        normalized_words.extend(w.split())
        
    # Hapus stopwords (tanpa menghiraukan negasi, sesuai dataset)
    words = [w for w in normalized_words if w not in stopwords and len(w) > 1]
    
    # TIDAK MENGGUNAKAN STEMMING (Sesuai dengan dataset ulasan_clean)
    
    cleaned = ' '.join(words).strip()
    return cleaned if cleaned else 'kosong'


# ── Model ────────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1TZshU6aeKerqa7sdzx0gqPVv0XUH3y3X"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        import gdown
        with st.spinner("Mengunduh model dari Google Drive, harap tunggu..."):
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                MODEL_PATH,
                quiet=False,
            )
    return joblib.load(MODEL_PATH)


def predict(text: str, model):
    clean = preprocess_text(text)
    # The pipeline expects an iterable of strings (like a list), not a DataFrame.
    # Passing a DataFrame causes TfidfVectorizer to vectorize the column names!
    idx = model.predict([clean])[0]
    return LABEL_MAP[idx], clean


# ── UI ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Klasifikasi Sentimen Telaga Sarangan",
        layout="centered",
    )

    # -- Minimal Custom CSS for structural tweaks --
    st.markdown("""
    <style>
    /* Hide Default Streamlit Elements */
    footer {visibility: hidden;}
    
    /* Clean up the text area */
    .stTextArea textarea {
        font-size: 1rem !important;
        border-radius: 8px !important;
    }
    
    /* Center text inside the result box */
    .result-title {
        text-align: center;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    .result-subtitle {
        text-align: center;
        margin-top: 0.5rem !important;
        opacity: 0.8;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # -- Header --
    st.title("Klasifikasi Sentimen Ulasan Wisata")
    st.markdown("Sistem analisis teks menggunakan algoritma Random Forest untuk menentukan sentimen ulasan pengunjung wisata **Telaga Sarangan**.")
    st.divider()

    # -- Sidebar --
    with st.sidebar:
        st.subheader("Informasi Sistem")
        st.write(
            "Aplikasi ini mengklasifikasikan sentimen ulasan wisata "
            "Telaga Sarangan ke dalam tiga kategori:\n"
            "- **Positif**\n"
            "- **Netral**\n"
            "- **Negatif**"
        )
        st.divider()
        st.subheader("Contoh Ulasan")
        
        examples = [
            "Telaga sarangan indah banget, udaranya sejuk dan pemandangan menakjubkan",
            "Macet parah, tukang parkir nembak harga mahal banget, kecewa",
            "berjalanlah sebentar danau menit berjalan mengelilinginya",
            "Tempatnya bersih dan nyaman, cocok buat liburan keluarga",
            "Jorok banget toiletnya, sampah dimana-mana, sangat mengecewakan",
            "terpenting kondisi kendaraan tenaga melewati jalanan terjal",
        ]
        
        for i, ex in enumerate(examples):
            if st.button(f"Contoh {i+1}", key=f"ex_{i}", help=ex, use_container_width=True):
                st.session_state["review_input"] = ex
                
        st.divider()
        st.subheader("Spesifikasi Model & Teknis")
        st.info(
            "**Metode dan Fitur:**\n"
            "- **Model:** Random Forest + SMOTE\n"
            "- **Fitur:** TF-IDF + Statistik Teks\n"
            "- **NLP:** Cleansing, Normalisasi Slang, Stopword Removal (tanpa Stemming)\n\n"
            "**Alasan Pemilihan Model:**\n"
            "- **Data Tidak Seimbang:** Ulasan negatif sangat minim (93 berbanding 1.333 netral dan 1.436 positif). SMOTE menyeimbangkan kelas data latih.\n"
            "- **SMOTE Terbaik:** Dibandingkan ADASYN, ROS, dan RUS, SMOTE menghasilkan F1-Macro (0.774) dan F1-Negatif (0.500) tertinggi.\n"
            "- **RF Tangguh:** Optimal dalam mengolah fitur campuran serta toleran terhadap bahasa slang."
        )

    # -- Load model --
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Sistem tidak dapat memuat model klasifikasi: {e}")
        st.stop()

    # -- Input --
    st.info("Silakan ketik teks ulasan pengunjung pada kolom di bawah ini, atau pilih contoh ulasan dari menu di sebelah kiri untuk mencoba sistem.")
    
    review = st.text_area(
        "Masukkan teks ulasan pengunjung:",
        value=st.session_state.get("review_input", ""),
        height=150,
        placeholder="Ketik ulasan mengenai wisata Telaga Sarangan di sini...",
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        clicked = st.button("Proses Analisis Sentimen", use_container_width=True, type="primary")

    # -- Result --
    if clicked:
        if not review or not review.strip():
            st.warning("Mohon masukkan teks ulasan sebelum melakukan analisis.")
        else:
            with st.spinner("Memproses analisis teks..."):
                label, cleaned = predict(review.strip(), model)

            st.divider()
            
            # Use native Streamlit container with border for modern look
            result_container = st.container(border=True)
            
            # Choose semantic colors that work in both light and dark mode
            if label == "Positif":
                color = "#22c55e" # emerald-500
            elif label == "Netral":
                color = "#94a3b8" # slate-400
            else:
                color = "#ef4444" # red-500
                
            with result_container:
                st.markdown(f"<h2 class='result-title' style='color: {color};'>Sentimen {label}</h2>", unsafe_allow_html=True)
                st.markdown("<p class='result-subtitle'>Berdasarkan klasifikasi algoritma Random Forest</p>", unsafe_allow_html=True)
            
            # Details expander
            with st.expander("Rincian Pemrosesan Teks (Preprocessing)"):
                st.markdown("**Teks Input:**")
                st.code(review.strip(), language="text")
                st.markdown("**Teks Hasil Pembersihan (Cleaned Text):**")
                st.code(cleaned, language="text")

    st.markdown("""
        <div style="text-align: center; margin-top: 5rem; padding-top: 1rem; border-top: 1px solid rgba(148, 163, 184, 0.2); color: rgba(148, 163, 184, 0.8); font-size: 0.875rem;">
            Sistem Klasifikasi Sentimen Wisata Telaga Sarangan © 2026
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
