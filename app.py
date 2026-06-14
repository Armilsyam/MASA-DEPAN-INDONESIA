import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import plotly.express as px

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

max_comments = st.sidebar.slider("Jumlah Maksimal Komentar:", min_value=20, max_value=500, value=150, step=10)

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
        optimis_words = {'optimis', 'cerah', 'maju', 'berhasil', 'mandiri', 'terjamin', 'modal', 'bagus', 'sehat', 'hebat', 'nyata', 'tangguh', 'mantap', 'setuju', 'dukung'}
        cemas_words = {'khawatir', 'suram', 'susah', 'meroket', 'bengkak', 'turun', 'berat', 'ancaman', 'sulit', 'utang', 'melonjak', 'beban', 'kecewa', 'mahal', 'rugi'}

        def analisa_prospek(teks):
            score = 0
            words = str(teks).lower().split()
            for w in words:
                if w in optimis_words: score += 1
                if w in cemas_words: score -= 1
            return 'Optimis (Positif)' if score > 0 else ('Cemas (Negatif)' if score < 0 else 'Netral/Ekspektatif')

        df_raw['Sentimen'] = df_raw['Komentar'].apply(analisa_prospek)
        
        # Ekstraksi komponen waktu untuk Heat Map harian/jam
        df_raw['Hari'] = df_raw['Tanggal'].dt.day_name()
        df_raw['Jam'] = df_raw['Tanggal'].dt.hour

        # Klasifikasi Kategori Otomatis untuk keperluan Stacked Bar & Sentiment Matrix
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
                fig, ax = plt.subplots()
                ax.imshow(wc_pos, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Data teks positif tidak mencukupi untuk Word Cloud.")

        with col_wc_neg:
            st.markdown("<h4 style='color:#e74c3c; text-align:center;'>Awan Kata: Narasi Kecemasan (Negatif)</h4>", unsafe_allow_html=True)
            neg_texts = dapatkan_clean_string(df_raw[df_raw['Sentimen'] == 'Cemas (Negatif)']['Komentar'])
            if len(neg_texts.strip()) > 5:
                wc_neg = WordCloud(width=600, height=350, background_color='white', colormap='Reds').generate(neg_texts)
                fig, ax = plt.subplots()
                ax.imshow(wc_neg, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Data teks negatif tidak mencukupi untuk Word Cloud.")
                
        st.markdown("---")

        # ====================================================================
        # COMPONENT 3: LINE GRAPH (TREN WAKTU)
        # ====================================================================
        st.subheader("📈 3. Monitoring Emosi Jangka Panjang (Line Graph)")
        df_sorted = df_raw.sort_values(by='Tanggal')
        if df_sorted['Tanggal'].dt.date.nunique() <= 1:
            df_sorted['Waktu_Grup'] = df_sorted['Tanggal'].dt.strftime('%H:%M')
        else:
            df_sorted['Waktu_Grup'] = df_sorted['Tanggal'].dt.date
            
        trend_df = df_sorted.groupby(['Waktu_Grup', 'Sentimen']).size().unstack(fill_value=0).reset_index()
        
        fig_line = px.line(
            trend_df, 
            x='Waktu_Grup', 
            y=[c for c in trend_df.columns if c != 'Waktu_Grup'],
            labels={'value': 'Volume Komentar', 'Waktu_Grup': 'Garis Waktu'},
            markers=True,
            color_discrete_map=COLOR_MAP
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("---")

        # ====================================================================
        # COMPONENT 4: STACKED BAR CHART
        # ====================================================================
        st.subheader("📊 4. Komparasi Sentimen per Klaster Isu (Stacked Bar Chart)")
        stacked_df = df_raw.groupby(['Kategori_Fokus', 'Sentimen']).size().unstack(fill_value=0).reset_index()
        
        fig_stacked = px.bar(
            stacked_df, 
            x='Kategori_Fokus', 
