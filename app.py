import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dasbor Sentimen RI Interaktif", layout="wide", page_icon="🇮🇩")

st.title("📊 Dasbor Analisis Sentimen & Tren Masa Depan Indonesia")
st.subheader("Peta Opini Netizen Jangka Panjang Terhadap Kepemimpinan Presiden Prabowo Subianto")
st.markdown("---")

# --- KONSISTENSI WARNA STATIS ---
COLOR_MAP = {'Optimis (Positif)': '#2ecc71', 'Cemas (Negatif)': '#e74c3c', 'Netral/Ekspektatif': '#95a5a6'}

# --- SIDEBAR INPUT & UTILITY ---
st.sidebar.header("📥 Pengaturan Komponen Crawling")
youtube_url = st.sidebar.text_input(
    "Masukkan Link Video YouTube:",
    value="https://youtube.com", 
    help="Salin tautan penuh dari video YouTube yang ingin Anda bedah komentarnya."
)

max_comments = st.sidebar.slider("Jumlah Maksimal Komentar:", min_value=20, max_value=10000, value=150, step=10)

# --- FUNGSI EXTRACT VIDEO ID ---
def get_video_id(url):
    regex = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(regex, url)
    return match.group(1) if match else None

# --- FUNGSI PARSING TANGGAL AMAN ---
def parse_youtube_date(time_str):
    try:
        return pd.to_datetime(time_str)
    except Exception:
        return pd.Timestamp.now()

# --- ENGINE CRAWLING DATA ---
@st.cache_data(show_spinner="Sedang mengunduh komentar dari YouTube... Mohon tunggu.")
def crawl_youtube_comments(url, limit):
    video_id = get_video_id(url)
    if not video_id:
        return pd.DataFrame()
    
    try:
        downloader = YoutubeCommentDownloader()
        comments_generator = downloader.get_comments(youtube_id=video_id, sort_by=0) 
        sliced_comments = islice(comments_generator, limit)
        
        data_list = []
        for comment in sliced_comments:
            raw_time = comment.get('time', '')
            parsed_time = parse_youtube_date(raw_time)
            
            data_list.append({
                'Tanggal': parsed_time,
                'Komentar': comment.get('text', ''),
                'Author': comment.get('author', 'Anonim'),
                'Likes': comment.get('votes', 0)
            })
            
        return pd.DataFrame(data_list)
        
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis saat crawling: {e}")
        return pd.DataFrame()

# --- PROSES EKSEKUSI DATA ---
if youtube_url:
    df_raw = crawl_youtube_comments(youtube_url, max_comments)
    
    if df_raw.empty:
        st.error("Gagal mengekstrak komentar. Pastikan format link YouTube benar dan video tersebut memiliki komentar publik.")
    else:
        # --- ENGINE ANALISIS SENTIMEN ---
        optimis_words = {'optimis', 'cerah', 'maju', 'berhasil', 'mandiri', 'terjamin', 'modal', 'bagus', 'sehat', 'hebat', 'nyata', 'tangguh', 'mantap', 'setuju', 'dukung', 'presiden'}
        cemas_words = {'khawatir', 'suram', 'susah', 'meroket', 'bengkak', 'turun', 'berat', 'ancaman', 'sulit', 'utang', 'melonjak', 'beban', 'kecewa', 'mahal', 'rugi', 'pajak'}

        def analisa_prospek(teks):
            score = 0
            words = str(teks).lower().split()
            for w in words:
                if w in optimis_words: score += 1
                if w in cemas_words: score -= 1
            return 'Optimis (Positif)' if score > 0 else ('Cemas (Negatif)' if score < 0 else 'Netral/Ekspektatif')

        df_raw['Sentimen'] = df_raw['Komentar'].apply(analisa_prospek)
        
        # Ekstraksi komponen waktu untuk visualisasi harian/jam
        df_raw['Hari'] = df_raw['Tanggal'].dt.day_name()
        df_raw['Jam'] = df_raw['Tanggal'].dt.hour

        # Klasifikasi Kategori Otomatis untuk Stacked Bar & Sentiment Matrix
        def kategorisasi_komentar(teks):
            teks = str(teks).lower()
            if any(k in teks for k in ['kebijakan', 'isi', 'prabowo', 'presiden', 'program', 'makan', 'gizi', 'pajak', 'negara', 'ekonomi']):
                return 'Kebijakan & Ekonomi'
            elif any(k in teks for k in ['kreator', 'admin', 'channel', 'gibran', 'tokoh', 'pemimpin', 'presiden']):
                return 'Performa Tokoh'
            else:
                return 'Pembahasan Umum'
        df_raw['Kategori_Fokus'] = df_raw['Komentar'].apply(kategorisasi_komentar)

        # --- METRIK INFORMASI UTAMA ---
        col1, col2, col3 = st.columns(3)
        total_data = len(df_raw)
        optimis_count = (df_raw['Sentimen'] == 'Optimis (Positif)').sum()
        cemas_count = (df_raw['Sentimen'] == 'Cemas (Negatif)').sum()

        with col1:
            st.metric("Total Komentar Diekstrak", f"{total_data} data")
        with col2:
            st.metric("Sentimen Masa Depan Optimis 👍", f"{(optimis_count/total_data)*100:.1f}%" if total_data > 0 else "0%")
        with col3:
            st.metric("Sentimen Masa Depan Cemas 😟", f"{(cemas_count/total_data)*100:.1f}%" if total_data > 0 else "0%")

        st.markdown("---")

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

        # ====================================================================
        # COMPONENT 1: PIE & DONUT CHART (RINGKASAN SEKEJAP)
        # ====================================================================
        st.subheader("🍩 1. Ringkasan Persentase Data (Donut Chart)")
        sentiment_counts = df_raw['Sentimen'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentimen', 'Jumlah']
        
        fig_donut = px.pie(
            sentiment_counts, 
            values='Jumlah', 
            names='Sentimen', 
            hole=0.6,
            color='Sentimen',
            color_discrete_map=COLOR_MAP
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("---")

        # ====================================================================
        # COMPONENT 2: DUAL WORD CLOUD (HIJAU VS MERAH)
        # ====================================================================
        st.subheader("☁️ 2. Komparasi Kata Kunci Dominan Audiens (Dual Word Cloud)")
        col_wc_pos, col_wc_neg = st.columns(2)
        
        stop_words_id = ['yang', 'dan', 'di', 'untuk', 'bisa', 'kita', 'pak', 'ini', 'itu', 'ada', 'dari', 'ya', 'ga', 'aja', 'dgn', 'video', 'he', 'si']
        
        def dapatkan_clean_string(list_komentar):
            teks_gabung = " ".join(list_komentar.astype(str).tolist())
            for stop in stop_words_id:
                teks_gabung = re.sub(r'\b' + stop + r'\b', '', teks_gabung, flags=re.IGNORECASE)
            return teks_gabung

        with col_wc_pos:
            st.markdown("<h4 style='color:#2ecc71; text-align:center;'>Awan Kata: Narasi Optimis (Positif)</h4>", unsafe_allow_html=True)
            pos_texts = dapatkan_clean_string(df_raw[df_raw['Sentimen'] == 'Optimis (Positif)']['Komentar'])
            if len(pos_texts.strip()) > 5:
                wc_pos = WordCloud(width=600, height=350, background_color='white', colormap='Greens').generate(pos_texts)
                fig_p, ax_p = plt.subplots()
                ax_p.imshow(wc_pos, interpolation='bilinear')
                ax_p.axis('off')
                st.pyplot(fig_p)
                plt.close(fig_p)
            else:
                st.info("Data teks positif tidak mencukupi untuk Word Cloud.")

        with col_wc_neg:
            st.markdown("<h4 style='color:#e74c3c; text-align:center;'>Awan Kata: Narasi Kecemasan (Negatif)</h4>", unsafe_allow_html=True)
            neg_texts = dapatkan_clean_string(df_raw[df_raw['Sentimen'] == 'Cemas (Negatif)']['Komentar'])
            if len(neg_texts.strip()) > 5:
                wc_neg = WordCloud(width=600, height=350, background_color='white', colormap='Reds').generate(neg_texts)
                fig_n, ax_n = plt.subplots()
                ax_n.imshow(wc_neg, interpolation='bilinear')
                ax_n.axis('off')
                st.pyplot(fig_n)
                plt.close(fig_n)
            else:
                st.info("Data teks negatif tidak mencukupi untuk Word Cloud.")

        st.markdown("---")

        # ====================================================================
        # COMPONENT 3 & 4: TREN WAKTU & STACKED BAR CHART
        # ====================================================================
        col_row2_left, col_row2_right = st.columns(2)

        with col_row2_left:
            st.subheader("📈 3. Linimasa Sentimen (Tren Waktu Harian)")
            # Mengelompokkan tren berdasarkan tanggal murni (Y-M-D) tanpa jam untuk grafik garis yang mulus
            df_raw['Tanggal_Hari'] = df_raw['Tanggal'].dt.date
            df_timeline = df_raw.groupby(['Tanggal_Hari', 'Sentimen']).size().reset_index(name='Jumlah')
            
            fig_line = px.line(
                df_timeline, x='Tanggal_Hari', y='Jumlah', color='Sentimen',
                color_discrete_map=COLOR_MAP, markers=True
            )
            fig_line.update_layout(hovermode="x unified", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_line, use_container_width=True)

        with col_row2_right:
            st.subheader("📊 4. Distribusi Sentimen per Kategori Fokus (Stacked Bar Chart)")
            df_bar = df_raw.groupby(['Kategori_Fokus', 'Sentimen']).size().reset_index(name='Jumlah')
            
            fig_bar = px.bar(
                df_bar, x='Kategori_Fokus', y='Jumlah', color='Sentimen',
                barmode='stack', color_discrete_map=COLOR_MAP
            )
            fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

   # ==========================================
# 7. COMPONENT 7: KESIMPULAN & SARAN MASUKAN DARI PREDIKSI
# ==========================================
st.markdown("---")
st.subheader("🔮 Kesimpulan Otomatis & Prediksi Rekomendasi Jangka Panjang")

# Kalkulasi ringkasan data yang valid menggunakan df_raw
total_komentar = len(df_raw)
total_pos = (df_raw['Sentimen'] == 'Optimis (Positif)').sum()
total_neg = (df_raw['Sentimen'] == 'Cemas (Negatif)').sum()
rasio_positif = (total_pos / total_komentar) * 100 if total_komentar > 0 else 0

# 1. TAMPILKAN METRIK PREDIKSI UTAMA
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label="Prediksi Stabilitas Sentimen Publik", 
        value=f"{rasio_positif:.1f}%", 
        delta="Kondisi Kondusif" if rasio_positif > 50 else "Butuh Intervensi Humas",
        delta_color="normal" if rasio_positif > 50 else "inverse"
    )

with kpi2:
    # Mencari kategori dengan keluhan tertinggi secara aman
    df_only_neg = df_raw[df_raw['Sentimen'] == 'Cemas (Negatif)']
    if not df_only_neg.empty:
        kategori_terburuk = df_only_neg.groupby('Kategori_Fokus').size().idxmax()
    else:
        kategori_terburuk = "Tidak Terdeteksi"
        
    st.metric(
        label="⚠️ Area Risiko Utama (Prediksi Resistensi)", 
        value=kategori_terburuk,
        delta="Fokus Kritik Tertinggi",
        delta_color="inverse"
    )

with kpi3:
    # Proyeksi dampak viralitas berdasarkan akumulasi likes komentar optimis
    total_likes_optimis = df_raw[df_raw['Sentimen'] == 'Optimis (Positif)']['Likes'].sum()
    st.metric(
        label="📈 Estimasi Efek Amplifikasi Dukungan", 
        value=f"+{total_likes_optimis} Interaksi", 
        delta="Skor Resonansi Organik"
    )

st.markdown("### 🎯 Saran Tindakan Nyata Berdasarkan Pemetaan Data:")

# 2. LOGIKA PEMBERIAN SARAN SECARA DINAMIS
col_saran1, col_saran2 = st.columns(2)

with col_saran1:
    st.info("💡 **Rekomendasi Strategi Narasi / Konten Ke Depan:**")
    st.write(
        f"Berdasarkan visualisasi Word Cloud, audiens merespons sangat positif ketika pembahasan menyentuh kata-kata "
        f"harapan pembangunan. Pertahankan penyajian data infografis konkret seputar program kerja nyata. "
        f"Gunakan momentum tren waktu saat ini untuk menggandakan kuantitas publikasi narasi optimis tersebut."
    )

with col_saran2:
    st.warning("🛠️ **Langkah Mitigasi Krisis & Isu Keluhan:**")
    st.write(
        f"Data grafik komparatif menunjukkan isu **{kategori_terburuk}** merupakan titik kumpul sentimen cemas terbesar. "
        f"Disarankan bagi kreator atau tim komunikasi terkait untuk segera merilis materi klarifikasi, edukasi kebijakan, "
        f"atau penyeimbang informasi guna meredam pergeseran opini negatif yang lebih meluas di area tersebut."
    )
