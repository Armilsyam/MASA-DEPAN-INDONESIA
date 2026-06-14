import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Scraper & Sentimen YouTube RI", layout="wide", page_icon="🇮🇩")

st.title("📊 Analisis Sentimen & Bertipok Masa Depan Indonesia")
st.subheader("Crawling Komentar Langsung dari Link YouTube Berita/Opini Presiden Prabowo Subianto")
st.markdown("---")

# --- SIDEBAR INPUT & UTILITY ---
st.sidebar.header("📥 Pengaturan Komponen Crawling")
youtube_url = st.sidebar.text_input(
    "Masukkan Link Video YouTube:",
    value="https://youtube.com", # Ganti dengan link video berita Prabowo
    help="Salin tautan penuh dari video YouTube yang ingin Anda bedah komentarnya."
)

max_comments = st.sidebar.slider("Jumlah Maksimal Komentar:", min_value=10, max_value=500, value=100, step=10)

# --- FUNGSI EXTRACT VIDEO ID ---
def get_video_id(url):
    # Regex untuk mengambil ID video dari berbagai format link YouTube
    regex = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(regex, url)
    return match.group(1) if match else None

# --- ENGINE CRAWLING DATA ---
@st.cache_data(show_spinner="Sedang mengunduh komentar dari YouTube... Mohon tunggu.")
def crawl_youtube_comments(url, limit):
    video_id = get_video_id(url)
    if not video_id:
        return pd.DataFrame()
    
    downloader = YoutubeCommentDownloader()
    # Mengambil generator komentar dari library youtube-comment-downloader
    comments_generator = downloader.get_comments_from_video(video_id, sort_by=0) # 0 = Populer, 1 = Terbaru
    
    # Ambil data sesuai limit batasan slider
    sliced_comments = islice(comments_generator, limit)
    
    data_list = []
    for comment in sliced_comments:
        data_list.append({
            'Tanggal': pd.to_datetime(comment['time']),
            'Komentar': comment['text'],
            'Author': comment['author'],
            'Likes': comment['votes']
        })
        
    return pd.DataFrame(data_list)

# --- PROSES EKSEKUSI DATA ---
if youtube_url:
    df_raw = crawl_youtube_comments(youtube_url, max_comments)
    
    if df_raw.empty:
        st.error("Gagal mengekstrak komentar. Pastikan format link YouTube benar dan video tersebut memiliki komentar publik.")
    else:
        # --- ENGINE ANALISIS SENTIMEN ---
        optimis_words = {'optimis', 'cerah', 'maju', 'berhasil', 'mandiri', 'terjamin', 'modal', 'bagus', 'sehat', 'hebat', 'nyata', 'tangguh', 'mantap', 'setuju'}
        cemas_words = {'khawatir', 'suram', 'susah', 'meroket', 'bengkak', 'turun', 'berat', 'ancaman', 'sulit', 'utang', 'melonjak', 'beban', 'kecewa', 'mahal'}

        def analisa_prospek(teks):
            score = 0
            words = str(teks).lower().split()
            for w in words:
                if w in optimis_words: score += 1
                if w in cemas_words: score -= 1
            return 'Optimis (Positif)' if score > 0 else ('Cemas (Negatif)' if score < 0 else 'Netral/Ekspektatif')

        df_raw['Sentimen'] = df_raw['Komentar'].apply(analisa_prospek)

        # --- METRIK UTAMA ---
        col1, col2, col3 = st.columns(3)
        total_data = len(df_raw)
        optimis_count = (df_raw['Sentimen'] == 'Optimis (Positif)').sum()
        cemas_count = (df_raw['Sentimen'] == 'Cemas (Negatif)').sum()

        with col1:
            st.metric("Total Komentar Sukses Dicrawling", f"{total_data} data")
        with col2:
            st.metric("Sentimen Masa Depan Optimis 👍", f"{(optimis_count/total_data)*100:.1f}%" if total_data > 0 else "0%")
        with col3:
            st.metric("Sentimen Masa Depan Cemas 😟", f"{(cemas_count/total_data)*100:.1f}%" if total_data > 0 else "0%")

        st.markdown("---")

        # --- VISUALISASI GRAFIK ---
        left_chart, right_chart = st.columns(2)

        with left_chart:
            st.subheader("📌 Distribusi Komparasi Sentimen")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df_raw, x='Sentimen', palette='coolwarm', ax=ax)
            plt.xticks(rotation=15)
            st.pyplot(fig)

        with right_chart:
            st.subheader("📈 Pola Tren Berdasarkan Waktu Data")
            # Mengurutkan berdasarkan tanggal untuk visualisasi deret waktu
            df_sorted = df_raw.sort_values(by='Tanggal')
            trend_df = df_sorted.groupby([df_sorted['Tanggal'].dt.date, 'Sentimen']).size().unstack(fill_value=0)
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for col in trend_df.columns:
                ax.plot(trend_df.index, trend_df[col], marker='o', label=col, linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(rotation=30)
            plt.legend()
            st.pyplot(fig)

        st.markdown("---")

        # --- KLASTER BERTIPOK (TOPIK UTAMA) ---
        st.subheader("🗂️ Temuan Ragam Isu Utama Video (Konsep Bertipok)")
        if len(df_raw) >= 5:
            # Menggunakan TF-IDF untuk menyaring stopword dasar Bahasa Indonesia
            stop_words_id = ['yang', 'dan', 'di', 'untuk', 'bisa', 'kita', 'pak', 'ini', 'itu', 'ada', 'dari', 'ya', 'ga', 'aja', 'dgn']
            vectorizer = TfidfVectorizer(max_features=500, stop_words=stop_words_id)
            tfidf = vectorizer.fit_transform(df_raw['Komentar'])
            
            # Membagi narasi menjadi 3 klaster topik utama
            nmf = NMF(n_components=3, random_state=42)
            nmf.fit(tfidf)
            feature_names = vectorizer.get_feature_names_out()
            
            t_col1, t_col2, t_col3 = st.columns(3)
            cols = [t_col1, t_col2, t_col3]
            
            for idx, topic in enumerate(nmf.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-4:]]
                with cols[idx]:
                    st.info(f"**Klaster Opini Ke-{idx+1}**\n\nKata Kunci Dominan: *{', '.join(top_words)}*")
        else:
            st.info("Jumlah narasi teks terlalu sedikit untuk memetakan klaster topik.")

        # --- PREDIKSI STRATEGIS ---
        st.markdown("---")
        st.subheader("🔮 Prediksi & Konklusi Respons Netizen")
        optimis_rate = (optimis_count / total_data) * 100 if total_data > 0 else 0

        if optimis_rate > 50:
            st.success(f"**PREDIKSI BERHASIL:** Video ini didominasi respons positif ({optimis_rate:.1f}%). Visi kebijakan masa depan yang dibahas dalam video YouTube ini diprediksi mendapat dukungan penuh publik secara digital.")
        elif optimis_rate == 0 and cemas_count == 0:
            st.warning("**PREDIKSI NETRAL:** Komentar didominasi oleh teks normatif/netral. Masyarakat digital cenderung bersikap memantau perkembangan situasi tanpa ekspresi emosi berlebih.")
        else:
            st.error(f"**PREDIKSI WASPADA:** Sentimen kecemasan/kritik tinggi. Narasi pada video ini berpotensi memicu gelombang kritik lebih besar di platform lain jika komunikasi publik pemerintah tidak segera diselaraskan.")

        # --- TABEL DATA HASIL SCRAPING ---
        with st.expander("Lihat Rincian Data Mentah Hasil Crawling"):
            st.dataframe(df_raw[['Tanggal', 'Author', 'Komentar', 'Likes', 'Sentimen']])
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Scraper & Sentimen YouTube RI", layout="wide", page_icon="🇮🇩")

st.title("📊 Analisis Sentimen & Bertipok Masa Depan Indonesia")
st.subheader("Crawling Komentar Langsung dari Link YouTube Berita/Opini Presiden Prabowo Subianto")
st.markdown("---")

# --- SIDEBAR INPUT & UTILITY ---
st.sidebar.header("📥 Pengaturan Komponen Crawling")
youtube_url = st.sidebar.text_input(
    "Masukkan Link Video YouTube:",
    value="https://youtube.com", # Ganti dengan link video berita Prabowo
    help="Salin tautan penuh dari video YouTube yang ingin Anda bedah komentarnya."
)

max_comments = st.sidebar.slider("Jumlah Maksimal Komentar:", min_value=10, max_value=500, value=100, step=10)

# --- FUNGSI EXTRACT VIDEO ID ---
def get_video_id(url):
    # Regex untuk mengambil ID video dari berbagai format link YouTube
    regex = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(regex, url)
    return match.group(1) if match else None

# --- ENGINE CRAWLING DATA ---
@st.cache_data(show_spinner="Sedang mengunduh komentar dari YouTube... Mohon tunggu.")
def crawl_youtube_comments(url, limit):
    video_id = get_video_id(url)
    if not video_id:
        return pd.DataFrame()
    
    downloader = YoutubeCommentDownloader()
    # Mengambil generator komentar dari library youtube-comment-downloader
    comments_generator = downloader.get_comments_from_video(video_id, sort_by=0) # 0 = Populer, 1 = Terbaru
    
    # Ambil data sesuai limit batasan slider
    sliced_comments = islice(comments_generator, limit)
    
    data_list = []
    for comment in sliced_comments:
        data_list.append({
            'Tanggal': pd.to_datetime(comment['time']),
            'Komentar': comment['text'],
            'Author': comment['author'],
            'Likes': comment['votes']
        })
        
    return pd.DataFrame(data_list)

# --- PROSES EKSEKUSI DATA ---
if youtube_url:
    df_raw = crawl_youtube_comments(youtube_url, max_comments)
    
    if df_raw.empty:
        st.error("Gagal mengekstrak komentar. Pastikan format link YouTube benar dan video tersebut memiliki komentar publik.")
    else:
        # --- ENGINE ANALISIS SENTIMEN ---
        optimis_words = {'optimis', 'cerah', 'maju', 'berhasil', 'mandiri', 'terjamin', 'modal', 'bagus', 'sehat', 'hebat', 'nyata', 'tangguh', 'mantap', 'setuju'}
        cemas_words = {'khawatir', 'suram', 'susah', 'meroket', 'bengkak', 'turun', 'berat', 'ancaman', 'sulit', 'utang', 'melonjak', 'beban', 'kecewa', 'mahal'}

        def analisa_prospek(teks):
            score = 0
            words = str(teks).lower().split()
            for w in words:
                if w in optimis_words: score += 1
                if w in cemas_words: score -= 1
            return 'Optimis (Positif)' if score > 0 else ('Cemas (Negatif)' if score < 0 else 'Netral/Ekspektatif')

        df_raw['Sentimen'] = df_raw['Komentar'].apply(analisa_prospek)

        # --- METRIK UTAMA ---
        col1, col2, col3 = st.columns(3)
        total_data = len(df_raw)
        optimis_count = (df_raw['Sentimen'] == 'Optimis (Positif)').sum()
        cemas_count = (df_raw['Sentimen'] == 'Cemas (Negatif)').sum()

        with col1:
            st.metric("Total Komentar Sukses Dicrawling", f"{total_data} data")
        with col2:
            st.metric("Sentimen Masa Depan Optimis 👍", f"{(optimis_count/total_data)*100:.1f}%" if total_data > 0 else "0%")
        with col3:
            st.metric("Sentimen Masa Depan Cemas 😟", f"{(cemas_count/total_data)*100:.1f}%" if total_data > 0 else "0%")

        st.markdown("---")

        # --- VISUALISASI GRAFIK ---
        left_chart, right_chart = st.columns(2)

        with left_chart:
            st.subheader("📌 Distribusi Komparasi Sentimen")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df_raw, x='Sentimen', palette='coolwarm', ax=ax)
            plt.xticks(rotation=15)
            st.pyplot(fig)

        with right_chart:
            st.subheader("📈 Pola Tren Berdasarkan Waktu Data")
            # Mengurutkan berdasarkan tanggal untuk visualisasi deret waktu
            df_sorted = df_raw.sort_values(by='Tanggal')
            trend_df = df_sorted.groupby([df_sorted['Tanggal'].dt.date, 'Sentimen']).size().unstack(fill_value=0)
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for col in trend_df.columns:
                ax.plot(trend_df.index, trend_df[col], marker='o', label=col, linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(rotation=30)
            plt.legend()
            st.pyplot(fig)

        st.markdown("---")

        # --- KLASTER BERTIPOK (TOPIK UTAMA) ---
        st.subheader("🗂️ Temuan Ragam Isu Utama Video (Konsep Bertipok)")
        if len(df_raw) >= 5:
            # Menggunakan TF-IDF untuk menyaring stopword dasar Bahasa Indonesia
            stop_words_id = ['yang', 'dan', 'di', 'untuk', 'bisa', 'kita', 'pak', 'ini', 'itu', 'ada', 'dari', 'ya', 'ga', 'aja', 'dgn']
            vectorizer = TfidfVectorizer(max_features=500, stop_words=stop_words_id)
            tfidf = vectorizer.fit_transform(df_raw['Komentar'])
            
            # Membagi narasi menjadi 3 klaster topik utama
            nmf = NMF(n_components=3, random_state=42)
            nmf.fit(tfidf)
            feature_names = vectorizer.get_feature_names_out()
            
            t_col1, t_col2, t_col3 = st.columns(3)
            cols = [t_col1, t_col2, t_col3]
            
            for idx, topic in enumerate(nmf.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-4:]]
                with cols[idx]:
                    st.info(f"**Klaster Opini Ke-{idx+1}**\n\nKata Kunci Dominan: *{', '.join(top_words)}*")
        else:
            st.info("Jumlah narasi teks terlalu sedikit untuk memetakan klaster topik.")

        # --- PREDIKSI STRATEGIS ---
        st.markdown("---")
        st.subheader("🔮 Prediksi & Konklusi Respons Netizen")
        optimis_rate = (optimis_count / total_data) * 100 if total_data > 0 else 0

        if optimis_rate > 50:
            st.success(f"**PREDIKSI BERHASIL:** Video ini didominasi respons positif ({optimis_rate:.1f}%). Visi kebijakan masa depan yang dibahas dalam video YouTube ini diprediksi mendapat dukungan penuh publik secara digital.")
        elif optimis_rate == 0 and cemas_count == 0:
            st.warning("**PREDIKSI NETRAL:** Komentar didominasi oleh teks normatif/netral. Masyarakat digital cenderung bersikap memantau perkembangan situasi tanpa ekspresi emosi berlebih.")
        else:
            st.error(f"**PREDIKSI WASPADA:** Sentimen kecemasan/kritik tinggi. Narasi pada video ini berpotensi memicu gelombang kritik lebih besar di platform lain jika komunikasi publik pemerintah tidak segera diselaraskan.")

        # --- TABEL DATA HASIL SCRAPING ---
        with st.expander("Lihat Rincian Data Mentah Hasil Crawling"):
            st.dataframe(df_raw[['Tanggal', 'Author', 'Komentar', 'Likes', 'Sentimen']])
