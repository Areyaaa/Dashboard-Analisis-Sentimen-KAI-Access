!pip install google-play-scraper Sastrawi wordcloud -q

import pandas as pd
import numpy as np
import re
import string
from google_play_scraper import reviews_all, Sort

# Preprocessing
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Visualisasi
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- Bagian 1: Scraping Data ---
def scrape_google_play_reviews(app_id, lang='id', country='id'):
    """
    Mengambil semua ulasan untuk aplikasi tertentu dari Google Play Store.
    """
    print(f"Mulai scraping ulasan untuk aplikasi: {app_id}")
    try:
        result = reviews_all(
            app_id,
            sleep_milliseconds=0,
            lang=lang,
            country=country,
            sort=Sort.NEWEST
        )

        df = pd.DataFrame(result)
        # Memilih kolom yang relevan
        df = df[['content', 'score']]
        # Mengganti nama kolom
        df.rename(columns={'content': 'ulasan', 'score': 'rating'}, inplace=True)

        # Simpan ke CSV
        nama_file = f'reviews_{app_id}.csv'
        df.to_csv(nama_file, index=False)
        print(f"Scraping selesai. Data disimpan di '{nama_file}'")
        return df, nama_file
    except Exception as e:
        print(f"Terjadi kesalahan saat scraping: {e}")
        return None, None

# --- Bagian 2: Preprocessing Teks ---

# Inisialisasi Sastrawi
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()

def cleaning_text(text):
    """Menghapus karakter khusus, angka, URL, dan spasi berlebih."""
    text = re.sub(r'http\S+', '', text)  # Hapus URL
    text = re.sub(r'@[^\s]+', '', text) # Hapus mention
    text = re.sub(r'#\w+', '', text) # Hapus hashtag
    text = re.sub(r'\d+', '', text) # Hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation)) # Hapus tanda baca
    text = text.strip() # Hapus spasi di awal dan akhir
    text = re.sub(r'\s+', ' ', text) # Hapus spasi berlebih
    return text

def case_folding(text):
    """Mengubah semua teks menjadi huruf kecil."""
    return text.lower()

def tokenize_text(text):
    """Memecah teks menjadi token (kata)."""
    return text.split()

def remove_stopwords(tokens):
    """Menghapus kata-kata umum (stopwords) dari daftar token."""
    # Sastrawi membutuhkan string, jadi kita gabungkan lalu proses
    text = ' '.join(tokens)
    return stopword_remover.remove(text).split()

def stem_text(tokens):
    """Mengubah setiap kata dalam token menjadi kata dasarnya (stemming)."""
    # Sastrawi membutuhkan string, jadi kita gabungkan lalu proses
    text = ' '.join(tokens)
    return stemmer.stem(text)

def preprocess_pipeline(text):
    """Menjalankan semua langkah preprocessing secara berurutan."""
    text = cleaning_text(text)
    text = case_folding(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    stemmed_text = stem_text(tokens)
    return stemmed_text

def demonstrate_preprocessing(sample_text):
    """
    Menampilkan input dan output dari setiap langkah preprocessing.
    """
    print("="*50)
    print("CONTOH PROSES PREPROCESSING")
    print("="*50)
    print(f"Teks Asli:\n'{sample_text}'\n")

    # 1. Cleaning
    cleaned = cleaning_text(sample_text)
    print(f"1. Cleaning Text:\n   Input: '{sample_text}'\n   Output: '{cleaned}'\n")

    # 2. Case Folding
    folded = case_folding(cleaned)
    print(f"2. Case Folding:\n   Input: '{cleaned}'\n   Output: '{folded}'\n")

    # 3. Tokenizing
    tokenized = tokenize_text(folded)
    print(f"3. Tokenizing:\n   Input: '{folded}'\n   Output: {tokenized}\n")

    # 4. Stopword Removal
    no_stopwords = remove_stopwords(tokenized)
    print(f"4. Stopword Removal:\n   Input: {tokenized}\n   Output: {no_stopwords}\n")

    # 5. Stemming
    stemmed_result = stem_text(no_stopwords)
    print(f"5. Stemming:\n   Input: {no_stopwords}\n   Output: '{stemmed_result}'\n")
    print("="*50)

# --- Bagian 3: Pemodelan dan Evaluasi ---

def plot_confusion_matrix(cm, labels, title):
    """Membuat plot untuk confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def generate_wordcloud(text_series, title):
    """Membuat dan menampilkan word cloud dari serangkaian teks."""
    text = ' '.join(text_series)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def run_analysis(df):
    """
    Menjalankan seluruh alur analisis sentimen dari preprocessing hingga evaluasi.
    """
    # Hapus baris dengan ulasan kosong
    df.dropna(subset=['ulasan'], inplace=True)
    df = df[df['ulasan'].str.strip() != '']

    # Tampilkan contoh preprocessing
    sample_review = df['ulasan'].iloc[0]
    demonstrate_preprocessing(sample_review)

    # Terapkan preprocessing ke seluruh dataset
    print("Menerapkan preprocessing ke seluruh dataset...")
    df['processed_text'] = df['ulasan'].apply(preprocess_pipeline)
    print("Preprocessing dataset selesai.")

    # Pemberian label sentimen
    # Rating 1, 2 -> Negatif (0)
    # Rating 4, 5 -> Positif (2)
    # Rating 3 -> Netral (1)
    df['sentimen'] = df['rating'].apply(lambda x: 1 if x == 3 else (0 if x <= 2 else 1))
    df_model = df[df['sentimen'] != -1].copy()

    if df_model.empty:
        print("Tidak ada data yang cukup untuk pemodelan setelah filtering sentimen.")
        return

    X = df_model['processed_text']
    y = df_model['sentimen']

    # Skema split data
    split_schemes = [
        {'test_size': 0.3, 'name': '70:30'},
        {'test_size': 0.2, 'name': '80:20'},
        {'test_size': 0.1, 'name': '90:10'}
    ]

    for scheme in split_schemes:
        print("\n" + "="*60)
        print(f"ANALISIS DENGAN SKEMA SPLIT DATA: {scheme['name']}")
        print("="*60)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=scheme['test_size'], random_state=42, stratify=y
        )

        # Ekstraksi Fitur dengan TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Pelatihan Model Naive Bayes
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)

        # Prediksi
        y_pred = model.predict(X_test_tfidf)

        # Evaluasi
        print(f"Classification Report (Skema {scheme['name']}):")
        print(classification_report(y_test, y_pred, target_names=['Negatif', 'Positif']))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, ['Negatif', 'Positif'], f"Confusion Matrix (Skema {scheme['name']})")

    # --- Bagian 4: Word Cloud ---
    print("\n" + "="*60)
    print("GENERATING WORD CLOUD")
    print("="*60)

    positif_text = df_model[df_model['sentimen'] == 1]['processed_text']
    negatif_text = df_model[df_model['sentimen'] == 0]['processed_text']

    if not positif_text.empty:
        generate_wordcloud(positif_text, 'Word Cloud Sentimen Positif')
    else:
        print("Tidak ada ulasan positif untuk membuat word cloud.")

    if not negatif_text.empty:
        print("Tidak ada ulasan negatif untuk membuat word cloud.")
    else:
        generate_wordcloud(negatif_text, 'Word Cloud Sentimen Negatif')


APP_ID = 'com.kai.kaiticketing'

nama_file_csv = f'reviews_{APP_ID}.csv'
try:
  ulasan_df = pd.read_csv(nama_file_csv)
  print(f"Data ulasan dimuat dari file '{nama_file_csv}'.")
except FileNotFoundError:
  print(f"File '{nama_file_csv}' tidak ditemukan. Memulai proses scraping baru.")
  ulasan_df, _ = scrape_google_play_reviews(APP_ID)

df = ulasan_df.copy()

df.dropna(subset=['ulasan'], inplace=True)
df = df[df['ulasan'].str.strip() != '']

# only 100 rows data selected
# df = df.sample(n=100, random_state=42)

sample_review = df['ulasan'].iloc[0]
demonstrate_preprocessing(sample_review)

print("Menerapkan preprocessing ke seluruh dataset...")
df['processed_text'] = df['ulasan'].apply(preprocess_pipeline)
print("Preprocessing dataset selesai.")

def rating_to_sentiment(rating):
    if rating == 1 or rating == 2:
        return 0  # Negatif
    elif rating == 4 or rating == 5:
        return 2  # Positif
    elif rating == 3:
        return 1
    else:
        return -1

# Pemberian label sentimen
# Rating 1, 2 -> Negatif (0)
# Rating 4, 5 -> Positif (2)
# Rating 3 -> Netral (1)
df['sentimen'] = df['rating'].apply(rating_to_sentiment)

df_model = df.copy()

if df_model.empty:
  print("Tidak ada data yang cukup untuk pemodelan setelah filtering sentimen.")

X = df_model['processed_text']
y = df_model['sentimen']

df

df_model['sentimen'].value_counts().plot(kind='bar')

split_schemes = [
    {'test_size': 0.3, 'name': '70:30'},
    {'test_size': 0.2, 'name': '80:20'},
    {'test_size': 0.1, 'name': '90:10'}
]

for scheme in split_schemes:
    print("\n" + "="*60)
    print(f"ANALISIS DENGAN SKEMA SPLIT DATA: {scheme['name']}")
    print("="*60)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=scheme['test_size'], random_state=42, stratify=y
    )

    # Ekstraksi Fitur dengan TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Pelatihan Model Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    # Prediksi
    y_pred = model.predict(X_test_tfidf)

    # Evaluasi
    print(f"Classification Report (Skema {scheme['name']}):")
    print(classification_report(y_test, y_pred, target_names=['Negatif','Netral','Positif']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ['Negatif','Netral','Positif'], f"Confusion Matrix (Skema {scheme['name']})")


df_model[df_model['sentimen'] == 0]

positif_text = df_model[df_model['sentimen'] == 2]['processed_text']
netral_text = df_model[df_model['sentimen'] == 1]['processed_text']
negatif_text = df_model[df_model['sentimen'] == 0]['processed_text']

generate_wordcloud(positif_text, 'Word Cloud Sentimen Positif')
generate_wordcloud(negatif_text, 'Word Cloud Sentimen Negatif')
generate_wordcloud(netral_text, 'Word Cloud Sentimen Netral')

