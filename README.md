# MASA-DEPAN-INDONESIA

Arsitektur & Alur Kerja Proyek[Crawling Data (TikTok/YT)] ➔ [Preprocessing Teks] ➔ [Analisis Sentimen (VADER/BERT)] ➔ [Pemodelan Topik (LDA)] ➔ [Analisis Tren & Prediksi]
1. Persiapan EnvironmentInstal pustaka Python yang dibutuhkan melalui terminal:bashpip install pandas numpy nltk sastrawi scikit-learn matplotlib seaborn
Gunakan kode dengan hati-hati.2. Script Python Lengkap (Proyek Sentimen & Topik)pythonimport pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- 1. SIMULASI DATA CRAWLING ---
# Contoh data hasil crawling komentar YouTube/TikTok/Facebook tentang Prabowo Subianto
data = {
    'tanggal': pd.to_date_range(start='2026-06-01', periods=10, freq='D').tolist() * 2,
    'komentar': [
        "Kebijakan ekonomi Pak Prabowo sangat berpihak pada rakyat kecil, maju terus!",
        "Harga bahan pokok kok malah naik terus di era pemerintahan sekarang? Kecewa.",
        "Program makan siang gratis dampaknya sangat terasa di sekolah anak saya.",
        "Rencana hilirisasi industri ini bagus untuk jangka panjang Indonesia.",
        "Masih banyak pengangguran, pemerintah harusnya fokus lapangan kerja baru.",
        "Diplomasi luar negeri Pak Presiden Prabowo membuat Indonesia disegani dunia.",
        "Bantuan sosial kurang tepat sasaran, tolong dievaluasi lagi kinerjanya.",
        "Infrastruktur di daerah luar Jawa mulai diperhatikan, mantap Presiden!",
        "Saya tidak melihat perubahan signifikan dalam 100 hari kerja ini.",
        "Pertahanan negara semakin kuat di bawah komando Prabowo Subianto.",
        "Sembako makin mahal, tolong perhatikan isi dompet masyarakat kecil.",
        "Kebijakan luar negeri yang tegas, pertemuannya dengan pemimpin dunia hebat.",
        "Anak-anak senang dengan program gizi gratis, sehat selalu pak.",
        "Isu korupsi masih jadi PR besar yang belum tuntas dibersihkan.",
        "Investasi asing masuk terus, semoga lapangan kerja makin terbuka luas.",
        "Langkah penegakan hukum terasa lambat dan tebang pilih.",
        "Swasembada pangan yang ditargetkan semoga cepat terealisasi.",
        "Gaji guru honorer kapan naik? Janjinya tolong ditepati.",
        "Sektor militer makin modern, bangga dengan kepemimpinan Prabowo.",
        "Pajak terus naik tapi fasilitas publik masih begitu-begitu saja."
    ]
}
df = pd.DataFrame(data)

# --- 2. PREPROCESSING TEKS (Pembersihan Data) ---
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus angka dan simbol
    tokens = word_tokenize(text)
    # Hapus stopword dan lakukan stemming singkat
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)

print("Memproses pembersihan teks...")
df['clean_komentar'] = df['komentar'].apply(clean_text)

# --- 3. ANALISIS SENTIMEN (Rule-Based Sederhana) ---
# Kamus kata positif dan negatif manual untuk bahasa Indonesia
pos_words = {'maju', 'bagus', 'hebat', 'terasa', 'senang', 'kuat', 'bangga', 'terasa', 'modern', 'terrealisasi'}
neg_words = {'kecewa', 'mahal', 'pengangguran', 'lambat', 'kurang', 'naik', 'korupsi', 'tuntut'}

def get_sentiment(text):
    score = 0
    words = text.split()
    for word in words:
        if word in pos_words: score += 1
        elif word in neg_words: score -= 1
    if score > 0: return 'Positif'
    elif score < 0: return 'Negatif'
    return 'Netral'

df['sentimen'] = df['clean_komentar'].apply(get_sentiment)

# --- 4. PEMODELAN TOPIK (BERTIPOK: Berita & Topik Utama) ---
vectorizer = CountVectorizer(max_features=1000)
tf_matrix = vectorizer.fit_transform(df['clean_komentar'])

# Menggunakan LDA untuk mengekstrak 3 topik utama
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(tf_matrix)

# Menampilkan kata kunci untuk setiap topik
words = vectorizer.get_feature_names_out()
print("\n--- Ekstraksi Bertipok (Topik Utama Opini) ---")
for idx, topic in enumerate(lda.components_):
    top_words = [words[i] for i in topic.argsort()[-4:]]
    print(f"Topik {idx+1}: {', '.join(top_words)}")

# --- 5. PREDIKSI & VISUALISASI TREN TEPAT WAKTU ---
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='sentimen', palette='viridis')
plt.title('Distribusi Sentimen Terhadap Presiden Prabowo Subianto')
plt.xlabel('Kategori Sentimen')
plt.ylabel('Jumlah Komentar')
plt.show()

# Kesimpulan Tren untuk Prediksi masa depan
positif_count = len(df[df['sentimen']=='Positif'])
negatif_count = len(df[df['sentimen']=='Negatif'])
print("\n--- Kesimpulan & Prediksi Kinerja ---")
if positif_count > negatif_count:
    print("Prediksi: Tren kepercayaan publik diprediksi tetap stabil/meningkat.")
    print("Faktor pendorong: Kepuasan pada program gizi, pertahanan, dan diplomasi.")
else:
    print("Prediksi: Potensi penurunan kepuasan publik jika isu ekonomi tidak diatasi.")
    print("Faktor pemicu: Sentimen negatif didominasi oleh isu harga barang dan pajak.")
Gunakan kode dengan hati-hati.Hubungan Konsep Bertipok dan PrediksiKonsep Bertipok (Berita & Topik Utama): Melalui algoritma LDA pada kode di atas, narasi besar netizen dipecah menjadi klaster isu spesifik. Misalnya, Topik 1 membahas ekonomi/harga pangan, Topik 2 membahas program gizi/sosial, dan Topik 3 membahas diplomasi/pertahanan.Prediksi Kebijakan & Elektabilitas: Dengan memantau pergeseran grafik sentimen harian (time-series), sistem dapat memprediksi kapan suatu isu akan memicu demonstrasi digital atau penurunan kepuasan publik sebelum dampaknya meluas ke dunia nyata.Jika Anda ingin mengembangkan proyek ini lebih jauh, beri tahu saya:Apakah Anda membutuhkan kode scraper otomatis (misalnya menggunakan yt-dlp atau API resmi) untuk mengambil data asli?Apakah Anda ingin meningkatkan akurasi sentimen menggunakan model AI (IndoBERT)?Apakah visualisasi trennya ingin diubah ke bentuk grafik garis berdasarkan waktu (time-series)?
