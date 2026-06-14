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

# --- CSS UNTUK HEADER STICKY YANG AMAN (TIDAK TERTUTUP SIDEBAR) ---
st.markdown(
    """
    <style>
    /* Menggunakan sticky agar posisi mengikuti container utama (aman dari sidebar) */
    .sticky-header {
        position: sticky;
        top: -30px; /* Menempel tepat di bawah top-bar bawaan streamlit */
        background-color: white;
        z-index: 99;
        padding: 15px 0px;
        margin-bottom: 20px;
        border-bottom: 2px solid #f0f2f6; /* Garis pembatas tipis yang elegan */
    }
    
    /* Menyesuaikan warna background di mode gelap */
    @media (prefers-color-scheme: dark) {
        .sticky-header {
            background-color: #0e1117;
            border-bottom: 2px solid #262730;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- MENAMPILKAN JUDUL DALAM BLOK STICKY ---
st.markdown(
    """
    <div class="sticky-header">
        <h1 style='margin:0; padding:0; font-size:2.2rem;'>📊 Dasbor Analisis Sentimen & Tren Masa Depan Indonesia</h1>
        <p style='margin:5px 0 0 0; padding:0; font-size:1.1rem; color:#666;'>Peta Opini Netizen Jangka Panjang Terhadap Kepemimpinan Presiden Prabowo Subianto</p>
    </div>
    """,
    unsafe_allow_html=True
)
# (Catatan: Hapus bagian `header-spacer` lama karena posisi sticky sudah otomatis mengatur jarak)

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
        # ====================================================================
        # COMPONENT 7: KESIMPULAN & REKOMENDASI KEBIJAKAN PEMERINTAHAN (RE-ENGINEERED)
        # ====================================================================
        st.markdown("---")
        st.subheader("🔮 Kesimpulan Otomatis & Analisis Prediktif Rekomendasi Pemerintahan")
        st.markdown(" Menyajikan ringkasan eksekutif dan rekomendasi taktis berbasis data untuk pengambil keputusan (Policy Makers).")

        # 1. Kalkulasi Parameter Parameter Utama
        total_komentar = len(df_raw)
        total_pos = (df_raw['Sentimen'] == 'Optimis (Positif)').sum()
        total_neg = (df_raw['Sentimen'] == 'Cemas (Negatif)').sum()
        rasio_optimis = (total_pos / total_komentar) * 100 if total_komentar > 0 else 0

        # Identifikasi kluster isu yang paling krusial bagi publik
        df_only_neg = df_raw[df_raw['Sentimen'] == 'Cemas (Negatif)']
        if not df_only_neg.empty:
            isu_risiko = df_only_neg.groupby('Kategori_Fokus').size().idxmax()
            jumlah_keluhan_isu_ini = df_only_neg[df_only_neg['Kategori_Fokus'] == isu_risiko].shape[0]
            persentase_keluhan_isu = (jumlah_keluhan_isu_ini / total_neg) * 100
        else:
            isu_risiko = "Tidak Terdeteksi"
            persentase_keluhan_isu = 0

        total_likes_optimis = df_raw[df_raw['Sentimen'] == 'Optimis (Positif)']['Likes'].sum()

        # 2. TAMPILKAN METRIK UTAMA BERBASIS KPI PEMERINTAHAN
        kpi_gov1, kpi_gov2, kpi_gov3 = st.columns(3)

        with kpi_gov1:
            # Mengukur tingkat penerimaan (Approval Rating) publik digital
            status_stabilitas = "Kondusif & Stabil" if rasio_optimis >= 55 else "Risiko Defisit Kepercayaan"
            st.metric(
                label="📊 Proyeksi Indeks Penerimaan Publik (Approval Rating)", 
                value=f"{rasio_optimis:.1f}%", 
                delta=status_stabilitas,
                delta_color="normal" if rasio_optimis >= 55 else "inverse"
            )

        with kpi_gov2:
            # Mengisolasi titik sumbat komunikasi publik terbesar
            st.metric(
                label="⚠️ Alarm Krisis (Titik Kritik Tertinggi)", 
                value=isu_risiko,
                delta=f"{persentase_keluhan_isu:.1f}% dari Total Komentar Negatif",
                delta_color="inverse"
            )

        with kpi_gov3:
            # Mengukur kekuatan 'organic amplifier' atau pendukung kebijakan di kolom komentar
            st.metric(
                label="📢 Skor Resonansi Narasi Positif (Aktivitas Likes)", 
                value=f"{total_likes_optimis} Reaksi", 
                delta="Dukungan Publik Organik"
            )

        # 3. LOGIKA FORMULASI REKOMENDASI DINAMIS BERDASARKAN HASIL DATA
        st.markdown("### 🎯 Lembar Rekomendasi Taktis Instansi Pemerintahan:")
        col_gov_saran1, col_gov_saran2 = st.columns(2)

        with col_gov_saran1:
            st.info("🏛️ **1. Rekomendasi Strategi Komunikasi Publik (Humas / Kemkominfo):**")
            
            # Logika dinamis untuk Humas berdasarkan tingkat Approval Rating
            if rasio_optimis < 40:
                st.write(
                    f"🚨 **STATUS DARURAT KOMUNIKASI.** Sentimen optimis berada di bawah angka 40%. "
                    f"Narasi publik saat ini didominasi oleh misinformasi atau ketidakpastian. "
                    f"**Tindakan:** Tim Humas Pemerintah harus segera melakukan *Counter-Narrative* massal. "
                    f"Hentikan sementara kampanye satu arah (Top-Down). Buka ruang dialog interaktif seperti "
                    f"Live Q&A bersama menteri atau tokoh kunci di YouTube/Media Sosial untuk menjawab keraguan netizen secara transparan."
                )
            elif 40 <= rasio_optimis < 60:
                st.write(
                    f"⚠️ **STATUS WASPADA TRANSISI.** Opini publik masih terbelah seimbang. "
                    f"Kelompok *Ekspektatif (Netral)* sangat mendominasi dan bisa bergeser menjadi negatif jika salah langkah. "
                    f"**Tindakan:** Masifkan penyusunan materi infografis dan video pendek (Shorts/TikTok) yang membedah keuntungan konkret dari kebijakan Presiden Prabowo. "
                    f"Fokuskan pada visualisasi alur manfaat langsung yang diterima masyarakat bawah."
                )
            else:
                st.write(
                    f"🟢 **STATUS AMAN & SUPORTIF.** Tingkat kepercayaan publik sangat tinggi. "
                    f"**Tindakan:** Kapitalisasi sentimen ini dengan memanfaatkan komentar-komentar positif yang memiliki "
                    f"banyak *likes* (`{total_likes_optimis} Reaksi`) untuk dijadikan testimoni digital resmi. "
                    f"Gunakan momentum ini untuk mengamplifikasi program kerja turunan tanpa hambatan resistensi digital."
                )

        with col_gov_saran2:
            st.warning("🛠️ **2. Rekomendasi Kebijakan Riil (Kementerian & Lembaga Terkait):**")
            
            # Logika dinamis berdasarkan rumpun isu yang paling banyak dikritik publik
            if isu_risiko == "Kebijakan & Ekonomi":
                st.write(
                    f"📈 **FOKUS ISU: EKONOMI & REGULASI NEGARA.** Netizen mengekspresikan kecemasan tinggi pada sub-topik "
                    f"anggaran, pajak, program pangan, atau stabilitas harga bahan pokok. "
                    f"**Tindakan Struktural:** Disarankan kepada Kementerian Koordinator Bidang Perekonomian dan Kementerian Keuangan "
                    f"untuk meninjau kembali variabel komunikasi dari penyesuaian tarif instrumen fiskal. Katup pengaman sosial "
                    f"(seperti subsidi tepat sasaran atau bantuan gizi) harus dipastikan terdistribusi 100% tepat waktu "
                    f"sebelum pengumuman regulasi makro dilakukan ke publik, guna menghindari gejolak penolakan sipil digital."
                )
            elif isu_risiko == "Performa Tokoh":
                st.write(
                    f"👤 **FOKUS ISU: LEGITIMASI & PERFORMA KEPEMIMPINAN.** Kritik berpusat pada penugasan kabinet atau figuritas penanggung jawab program. "
                    f"**Tindakan Struktural:** Diperlukan penegakan KPI (Key Performance Indicator) kementerian yang lebih terbuka kepada publik. "
                    f"Presiden disarankan memperkuat penegasan komitmen pemberantasan kebocoran anggaran negara dan hukum secara periodik "
                    f"untuk menjaga integritas serta wibawa politik pemerintahan baru di mata netizen."
                )
            else:
                st.write(
                    f"💬 **FOKUS ISU: PEMBAHASAN UMUM & DINAMIKA SOSIAL.** Keluhan tersebar di berbagai aspek non-struktural. "
                    f"**Tindakan Struktural:** Tetap lakukan monitoring berkala menggunakan alat ukur analitik ini. "
                    f"Secara umum, masyarakat masih memantau proses implementasi tahap awal, sehingga fokus utama "
                    f"adalah menjaga stabilitas pelayanan publik dasar agar tidak terjadi blunder operasional di lapangan."
                )
