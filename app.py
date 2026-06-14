import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA WARNA
# ==========================================
st.set_page_config(page_title="YouTube Sentiment Dashboard", layout="wide")

# Palet warna konsisten sesuai instruksi
COLOR_MAP = {
    'Positif': '#2ecc71', # Hijau
    'Netral': '#95a5a6',  # Abu-abu
    'Negatif': '#e74c3c'  # Merah
}

st.title("📊 YouTube Comment Sentiment Dashboard")
st.markdown("Dasbor interaktif tingkat lanjut untuk memetakan emosi dan topik audiens secara mendalam.")

# ==========================================
# 2. GENERATOR DATA SIMULASI (DUMMY DATA)
# ==========================================
@st.cache_data
def load_dummy_data():
    np.random.seed(42)
    dates = pd.date_range(start="2026-06-01", periods=14, freq='D')
    categories = ['Konten Edukasi', 'Kualitas Audio', 'Durasi Video', 'Performa Kreator']
    hours = list(range(0, 24, 4))
    
    data_list = []
    for date in dates:
        for cat in categories:
            for hour in hours:
                # Membuat acakan volume sentimen
                pos = np.random.randint(5, 30)
                neu = np.random.randint(2, 15)
                neg = np.random.randint(1, 20) if cat != 'Kualitas Audio' else np.random.randint(15, 40)
                
                data_list.append({'Tanggal': date, 'Jam': f"{hour:02d}:00", 'Kategori': cat, 'Sentimen': 'Positif', 'Jumlah': pos})
                data_list.append({'Tanggal': date, 'Jam': f"{hour:02d}:00", 'Kategori': cat, 'Sentimen': 'Netral', 'Jumlah': neu})
                data_list.append({'Tanggal': date, 'Jam': f"{hour:02d}:00", 'Kategori': cat, 'Sentimen': 'Negatif', 'Jumlah': neg})
                
    return pd.DataFrame(data_list)

df = load_dummy_data()

# Teks simulasi khusus untuk Word Cloud
text_positif = "keren bermanfaat bagus mendidik cerdas mantap suka membantu informatif menarik seru clear jelas"
text_negatif = "kecewa jelek bising kresek lambat membosankan buruk kurang bingung parah patah rusak kecewa"

# ==========================================
# 3. SIDEBAR / FILTER INTERAKTIF
# ==========================================
st.sidebar.header("🎛️ Filter Analisis")
selected_category = st.sidebar.multiselect(
    "Pilih Kategori Topik:", 
    options=df['Kategori'].unique(), 
    default=df['Kategori'].unique()
)

# Filter Data Berdasarkan Pilihan
df_filtered = df[df['Kategori'].isin(selected_category)]

# ==========================================
# 4. ROW 1: PIE/DONUT CHART & WORD CLOUD
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 Ringkasan Sentimen Utama")
    # Agregasi data untuk Pie/Donut Chart
    df_pie = df_filtered.groupby('Sentimen')['Jumlah'].sum().reset_index()
    
    fig_donut = px.pie(
        df_pie, values='Jumlah', names='Sentimen', hole=0.5,
        color='Sentimen', color_discrete_map=COLOR_MAP
    )
    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
    fig_donut.update_layout(showlegend=False, height=350, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    st.subheader("☁️ Word Cloud Kontras (Positif vs Negatif)")
    w_col1, w_col2 = st.columns(2)
    
    with w_col1:
        st.markdown("<p style='color:#2ecc71; font-weight:bold; text-align:center;'>🟢 Kata Kunci Positif</p>", unsafe_allow_html=True)
        wordcloud_pos = WordCloud(background_color="white", colormap="Greens", width=400, height=300).generate(text_positif)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(wordcloud_pos, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig, clear_figure=True)
        
    with w_col2:
        st.markdown("<p style='color:#e74c3c; font-weight:bold; text-align:center;'>🔴 Kata Kunci Negatif</p>", unsafe_allow_html=True)
        wordcloud_neg = WordCloud(background_color="white", colormap="Reds", width=400, height=300).generate(text_negatif)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(wordcloud_neg, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig, clear_figure=True)

# ==========================================
# 5. ROW 2: SENTIMENT TIMELINE & STACKED BAR CHART
# ==========================================
col3, col4 = st.columns(2)

with col3:
    st.subheader("📈 Sentiment Timeline (Tren Waktu)")
    df_timeline = df_filtered.groupby(['Tanggal', 'Sentimen'])['Jumlah'].sum().reset_index()
    
    fig_line = px.line(
        df_timeline, x='Tanggal', y='Jumlah', color='Sentimen',
        color_discrete_map=COLOR_MAP, markers=True
    )
    fig_line.update_layout(height=350, hovermode="x unified", margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_line, use_container_width=True)

with col4:
    st.subheader("📊 Perbandingan Berdasarkan Kategori")
    df_bar = df_filtered.groupby(['Kategori', 'Sentimen'])['Jumlah'].sum().reset_index()
    
    fig_bar = px.bar(
        df_bar, x='Kategori', y='Jumlah', color='Sentimen',
        barmode='stack', color_discrete_map=COLOR_MAP
    )
    fig_bar.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# 6. ROW 3: HEAT MAP & SENTIMENT MATRIX
# ==========================================
st.markdown("---")
col5, col6 = st.columns(2)

with col5:
    st.subheader("🗺️ Heat Map: Waktu Keluhan (Sentimen Negatif)")
    st.caption("Memetakan jam-jam rawan kemunculan sentimen negatif audiens.")
    
    df_neg = df_filtered[df_filtered['Sentimen'] == 'Negatif']
    df_heatmap = df_neg.groupby(['Jam', 'Kategori'])['Jumlah'].sum().unstack(fill_value=0)
    
    fig_heat = px.imshow(
        df_heatmap, 
        labels=dict(x="Kategori", y="Jam Kerja/Waktu", color="Jumlah Komentar"),
        x=df_heatmap.columns,
        y=df_heatmap.index,
        color_continuous_scale="Reds"
    )
    fig_heat.update_layout(height=380, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_heat, use_container_width=True)

with col6:
    st.subheader("🎛️ Sentiment Matrix (Sub-Topik Spesifik)")
    st.caption("Bedah perbandingan volume emosi antar fitur atau bagian video.")
    
    df_matrix = df_filtered.groupby(['Kategori', 'Sentimen'])['Jumlah'].sum().unstack(fill_value=0)
    # Kalkulasi rasio kepuasan (Positif / Negatif) sebagai nilai matriks
    df_matrix['Rasio Kepuasan'] = (df_matrix['Positif'] / (df_matrix['Negatif'] + 1)).round(2)
    
    fig_matrix = go.Figure(data=go.Heatmap(
        z=df_matrix['Rasio Kepuasan'],
        x=['Rasio Kepuasan (Positif vs Negatif)'],
        y=df_matrix.index,
        colorscale='RdYlGn', # Merah ke Hijau
        text=df_matrix['Rasio Kepuasan'],
        texttemplate="%{text}x Lipat",
        hoverongaps = False
    ))
    fig_matrix.update_layout(height=380, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_matrix, use_container_width=True)
# ==========================================
# 7. ROW 4: KESIMPULAN & SARAN MASUKAN PREDIKSI
# ==========================================
st.markdown("---")
st.subheader("🔮 Kesimpulan Otomatis & Prediksi Rekomendasi")

# Kalkulasi ringkasan data untuk logika inferensi/prediksi
total_pos = df_filtered[df_filtered['Sentimen'] == 'Positif']['Jumlah'].sum()
total_neg = df_filtered[df_filtered['Sentimen'] == 'Negatif']['Jumlah'].sum()
total_komentar = df_filtered['Jumlah'].sum()
rasio_positif = (total_pos / total_komentar) * 100 if total_komentar > 0 else 0

# 1. TAMPILKAN METRIK PREDIKSI UTAMA
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label="Prediksi Skor Kesehatan Channel", 
        value=f"{rasio_positif:.1f}%", 
        delta="Sangat Sehat" if rasio_positif > 60 else "Butuh Evaluasi",
        delta_color="normal" if rasio_positif > 60 else "inverse"
    )

with kpi2:
    # Mencari kategori dengan keluhan tertinggi
    kategori_terburuk = df_filtered[df_filtered['Sentimen'] == 'Negatif'].groupby('Kategori')['Jumlah'].sum().idxmax()
    st.metric(
        label="⚠️ Fokus Utama Perbaikan (Prediksi Risiko)", 
        value=kategori_terburuk,
        delta="Keluhan Tertinggi",
        delta_color="inverse"
    )

with kpi3:
    # Menghitung estimasi pertumbuhan penonton berdasarkan sentimen positif
    prediksi_subscribers = int(total_pos * 0.05) # Asumsi 5% komentator positif akan subscribe
    st.metric(
        label="📈 Proyeksi Konversi Subscriber Baru", 
        value=f"+{prediksi_subscribers} User", 
        delta="Berdasarkan Sentimen Positif"
    )

st.markdown("### 🎯 Saran Tindakan Nyata (Actionable Insights):")

# 2. LOGIKA PEMBERIAN SARAN SECARA DINAMIS
col_saran1, col_saran2 = st.columns(2)

with col_saran1:
    st.info("💡 **Rekomendasi Konten Selanjutnya:**")
    st.write(
        f"Berdasarkan tingginya kata kunci positif pada visualisasi Word Cloud, penonton sangat menyukai elemen "
        f"**Edukasi dan Gaya Penyampaian** Anda. Untuk video berikutnya, pertahankan gaya penjelasan yang lugas "
        f"dan buatlah sekuel atau bagian kedua dari topik video ini guna mempertahankan keterikatan (*engagement*) audiens."
    )

with col_saran2:
    st.warning("🛠️ **Rencana Perbaikan Teknis:**")
    st.write(
        f"Data dari Heat Map menunjukkan adanya penumpukan sentimen negatif pada kategori **{kategori_terburuk}**. "
        f"Penonton banyak mengeluhkan masalah teknis pada jam-jam utama penayangan video. "
        f"Disarankan untuk melakukan pengecekan ulang (QC) kualitas sebelum publikasi, atau berikan pin komentar berisi permohonan maaf dan solusi instan."
    )
