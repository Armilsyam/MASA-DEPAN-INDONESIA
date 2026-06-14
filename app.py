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
st.set_page_config(page_title="Scraper & Sentimen YouTube RI", layout="wide", page_icon="🇮🇩")

st.title("📊 Analisis Sentimen & Bertipok Masa Depan Indonesia")
st.subheader("Crawling Komentar Langsung dari Link YouTube Berita/Opini Presiden Prabowo Subianto")
st.markdown("---")

# --- SIDEBAR INPUT & UTILITY ---
st.sidebar.header("📥 Pengaturan Komponen Crawling")
youtube_url = st.sidebar.text_input(
    "Masukkan Link Video YouTube:",
    value="https://youtube.com", 
    help="Salin tautan penuh dari video YouTube yang ingin Anda bedah komentarnya."
)

max_comments = st.sidebar.slider("Jumlah Maksimal Komentar:", min_value=10, max_value=500, value=100, step=10)

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

        # ====================================================================
        # IDE 1 & 2: DONUT CHART (PEMBUKA) & WORD CLOUD (BEDAH KATA UTAMA)
        # ====================================================================
        col_donut, col_wordcloud = st.columns(2)

        with col_donut:
            st.subheader("🍩 1. Kondisi Umum Video (Donut Chart)")
            sentiment_counts = df_raw['Sentimen'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentimen', 'Jumlah']
            
            # Skema warna khusus: Hijau untuk Optimis, Merah untuk Cemas, Abu-abu untuk Netral
            color_map = {'Optimis (Positif)': '#2ecc71', 'Cemas (Negatif)': '#e74c3c', 'Netral/Ekspektatif': '#95a5a6'}
            
            fig_donut = px.pie(
                sentiment_counts, 
                values='Jumlah', 
                names='Sentimen', 
                hole=0.5,
                color='Sentimen',
                color_discrete_map=color_map
            )
            fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_wordcloud:
            st.subheader("☁️ 2. Kata Kunci Dominan Penonton (Word Cloud)")
            semua_teks = " ".join(df_raw['Komentar'].astype(str).tolist())
            
            # Membersihkan teks dari kata umum agar Word Cloud bermakna
            for stop in ['yang', 'dan', 'di', 'untuk', 'bisa', 'kita', 'pak', 'ini', 'itu', 'ada', 'dari', 'ya', 'ga', 'aja', 'dgn', 'video']:
                semua_teks = re.sub(r'\b' + stop + r'\b', '', semua_teks, flags=re.IGNORECASE)
            
            if len(semua_teks.strip()) > 0:
                wordcloud = WordCloud(width=800, height=450, background_color='white', colormap='viridis').generate(semua_teks)
                fig_wc, ax_wc = plt.subplots(figsize=(8, 4.5))
                ax_wc.imshow(wordcloud, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
            else:
                st.info("Teks tidak cukup untuk membangun Word Cloud.")

        st.markdown("---")

        # ====================================================================
        # IDE 3: DINAMIKA LEWAT LINE GRAPH (TIME-SERIES)
        # ====================================================================
        st.subheader("📈 3. Dinamika Lonjakan Tren Opini (Line Graph)")
        df_sorted = df_raw.sort_values(by='Tanggal')
        
        if df_sorted['Tanggal'].dt.date.nunique() <= 1:
            df_sorted['Waktu_Grup'] = df_sorted['Tanggal'].dt.strftime('%H:%M')
        else:
            df_sorted['Waktu_Grup'] = df_sorted['Tanggal'].dt.date
            
        trend_df = df_sorted.groupby(['Waktu_Grup', 'Sentimen']).size().unstack(fill_value=0).reset_index()
        
        # Plot menggunakan Plotly Line agar interaktif (bisa di-hover untuk melihat waktu tepatnya)
        fig_line = px.line(
            trend_df, 
            x='Waktu_Grup', 
            y=trend_df.columns[1:], 
            labels={'value': 'Jumlah Komentar', 'Waktu_Grup': 'Waktu Analisis'},
            markers=True
        )
        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("---")

        # ====================================================================
        # IDE 4: VALIDASI DENGAN BAR CHART KATEGORI BESAR
        # ====================================================================
        st.subheader("📊 4. Validasi Fokus Utama Komentar Penonton (Bar Chart)")
        
        # Rule-based sederhana untuk mengelompokkan komentar ke dalam 3 kategori besar keinginan user
        def kategorisasi_komentar(teks):
            teks = str(teks).lower()
            if any(k in teks for k in ['kebijakan', 'isi', 'prabowo', 'presiden', 'program', 'makan', 'gizi', 'pajak', 'negara', 'ekonomi']):
                return 'Isi Konten / Kebijakan Negara'
            elif any(k in teks for k in ['kreator', 'admin', 'channel', 'gibran', 'tokoh', 'pemimpin', 'presiden']):
                return 'Performa Tokoh / Kreator'
            else:
                return 'Kualitas Video / Pembahasan Umum'
                
        df_raw['Kategori_Fokus'] = df_raw['Komentar'].apply(kategorisasi_komentar)
        cat_counts = df_raw['Kategori_Fokus'].value_counts().reset_index()
        cat_counts.columns = ['Kategori Fokus', 'Jumlah']
        
        fig_bar = px.bar(
            cat_counts, 
            x='Jumlah', 
            y='Kategori Fokus', 
            orientation='h', 
            color='Kategori Fokus',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # ====================================================================
        # IDE 5: KESIMPULAN AKHIR MELALUI TREEMAP TOPIC WITH REAL COMMENTS
        # ====================================================================
        st.subheader("🗺️ 5. Peta Kesimpulan Topik & Contoh Komentar Riil (TreeMap)")
        
        # Membuat label ringkasan komentar singkat (max 40 karakter) sebagai contoh nyata di dalam kotak TreeMap
        df_raw['Ringkasan_Komentar'] = df_raw['Komentar'].apply(lambda x: str(x)[:40] + '...' if len(str(x)) > 40 else str(x))
        
        # Menghitung count agregat agar plotly bisa merender ukuran bidang kotak secara proporsional
        df_raw['Count'] = 1
        
        fig_treemap = px.treemap(
            df_raw, 
            path=['Kategori_Fokus', 'Sentimen', 'Ringkasan_Komentar'], 
            values='Count',
