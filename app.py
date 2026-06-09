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

from sklearn.base import BaseEstimator, TransformerMixin
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ── Configuration ────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
LABEL_MAP = {0: "Negatif", 1: "Netral", 2: "Positif"}
USE_STEMMING = True

NEGATION_PREFIXES = (
    'tidak', 'tak', 'bukan', 'jangan', 'kurang', 'belum', 'gak', 'ga', 'nggak'
)

POS_WORDS = {
    'bagus','indah','nyaman','sejuk','keren','mantap','asyik','recommended',
    'bersih','suka','ramah','seru','fresh','adem','luar',
    'cantik','menawan','menarik','asri','natural','alami','jernih',
    'puas','senang','bahagia','tenang','damai','amazing','beautiful',
    'lezat','enak','murah','terjangkau','worth','worthit',
    'josss','joss','top','hebat','luar biasa','istimewa','eksotis',
    'betah','cocok','pas','oke','ok','nice','good','great',
    'romantis','menyenangkan','dingin','seger','segar',
    'favorit','best','sempurna','memuaskan','apresiasi',
}
NEG_WORDS = {
    'mahal','macet','kotor','kecewa','parah','buruk','ruwet','bosan',
    'bau','jelek','kurang','jauh','antri','penuh','semrawut','tidak',
    'kumuh','jorok','rusak','bahaya','berbahaya','licin','sempit',
    'mengecewakan','payah','sampah','banjir','longsor','sepi',
    'tipu','penipu','pungli','bohong','nakal','kasar',
    'panas','gerah','becek','berlubang','ancur','hancur',
    'lambat','lama','susah','sulit','ribet','repot',
    'mengantri','antre','malas','males','capek','capai',
    'overpriced','kemahalan','murahan',
    'ngeri','takut','seram','horor','gelap','gersang',
}
NEU_WORDS = {
    'cukup','standar','lumayan','biasa','ramai','weekend','jalan','parkir',
    'tiket','harga','naik','turun','telaga','sarangan','wisata','pengunjung',
    'tempat','lokasi','area','kawasan','hotel','villa','penginapan',
    'restoran','warung','kuda','speedboat','boat','perahu',
    'gunung','danau','air','pohon','hutan','alam',
    'libur','liburan','jalan-jalan','piknik','rekreasi',
    'keluarga','anak','teman','rombongan','foto','selfie',
    'masuk','keluar','buka','tutup','jam',
}

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


class LexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract lexicon-based sentiment features from cleaned text."""
    def __init__(self, pos_words=None, neg_words=None, neu_words=None):
        self.pos_words = pos_words
        self.neg_words = neg_words
        self.neu_words = neu_words

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pw = self.pos_words if self.pos_words is not None else POS_WORDS
        nw = self.neg_words if self.neg_words is not None else NEG_WORDS
        nuw = self.neu_words if self.neu_words is not None else NEU_WORDS
        rows = []
        for text in X.astype(str):
            tokens = text.split()
            total = max(len(tokens), 1)
            pos_c = neg_c = neu_c = 0
            for w in tokens:
                if '_' in w and w.split('_', 1)[0] in NEGATION_PREFIXES:
                    neg_c += 1
                    continue
                if w in pw:
                    pos_c += 1
                if w in nw:
                    neg_c += 1
                if w in nuw:
                    neu_c += 1
            rows.append({
                'lex_pos_count': pos_c, 'lex_neg_count': neg_c,
                'lex_neu_count': neu_c,
                'lex_pos_ratio': pos_c / total,
                'lex_neg_ratio': neg_c / total,
                'lex_neu_ratio': neu_c / total,
                'lex_polarity': pos_c - neg_c,
                'lex_subjectivity': pos_c + neg_c,
                'lex_total_hits': pos_c + neg_c + neu_c,
                'lex_hit_ratio': (pos_c + neg_c + neu_c) / total,
            })
        return pd.DataFrame(rows)


# ── NLP tools ────────────────────────────────────────────────────────
@st.cache_resource
def _load_nlp():
    stemmer = StemmerFactory().create_stemmer()
    sw = set(StopWordRemoverFactory().get_stop_words()) - set(NEGATION_PREFIXES)
    return stemmer, sw


def preprocess_text(text: str) -> str:
    stemmer, stopwords = _load_nlp()
    text = str(text).lower()
    text = re.sub(
        r'\b(tidak|tak|bukan|jangan|kurang|belum|gak|ga|nggak)\s+(\w+)',
        r'\1_\2', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@[\w]+', '', text)
    text = re.sub(r'#[\w]+', '', text)
    text = re.sub(r'\d+', '', text)
    p = string.punctuation.replace('_', '')
    text = text.translate(str.maketrans('', '', p))
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [SLANG_MAP.get(w, w) for w in text.split()]
    words = [w for w in words if w not in stopwords and len(w) > 1]
    if USE_STEMMING:
        neg = set(NEGATION_PREFIXES)
        processed = []
        for w in words:
            if '_' in w:
                pref, suf = w.split('_', 1)
                if pref in neg and suf:
                    processed.append(f"{pref}_{stemmer.stem(suf)}")
                    continue
            processed.append(stemmer.stem(w))
        words = processed
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
    df = pd.DataFrame({'raw': [text], 'clean': [clean]})
    idx = model.predict(df)[0]
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
            "Biasa aja sih tempatnya, standar wisata air pada umumnya",
            "Tempatnya bersih dan nyaman, cocok buat liburan keluarga",
            "Jorok banget toiletnya, sampah dimana-mana, sangat mengecewakan",
        ]
        
        for i, ex in enumerate(examples):
            if st.button(f"Contoh {i+1}", key=f"ex_{i}", help=ex, use_container_width=True):
                st.session_state["review_input"] = ex
                
        st.divider()
        st.subheader("Spesifikasi Model & Teknis")
        st.info(
            "**Metode dan Fitur:**\n"
            "- **Model:** Random Forest + SMOTE\n"
            "- **Fitur:** TF-IDF + Lexicon + Statistik Teks\n"
            "- **NLP:** Cleansing, Normalisasi Slang, Stemming\n\n"
            "**Alasan Pemilihan Model:**\n"
            "- **Data Tidak Seimbang:** Ulasan negatif sangat minim (93 berbanding 1.333 netral dan 1.436 positif). SMOTE menyeimbangkan kelas data latih.\n"
            "- **SMOTE Terbaik:** Dibandingkan ADASYN, ROS, dan RUS, SMOTE menghasilkan F1-Macro (0.774) dan F1-Negatif (0.500) tertinggi.\n"
            "- **RF Tangguh:** Optimal dalam mengolah fitur campuran serta toleran terhadap bahasa slang.\n"
            "- **Dampak Lexicon:** F1-Macro meningkat 11.7% (dari 0.692 menjadi 0.773) dan F1-Negatif meningkat 69.6% (dari 0.286 menjadi 0.485)."
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
