from flask import Flask, render_template, jsonify, request
import pandas as pd
from collections import Counter
import re
import os

app = Flask(__name__)

# =======================
# Konfigurasi
# =======================
CSV_PATH = "data/reviews_com.kai.kaiticketing.csv"

STOPWORDS = {
    "yang", "dan", "di", "ke", "untuk", "pada", "dengan", "dari", "itu", "ini", "ya",
    "tidak", "sudah", "ada", "karena", "jadi", "saja", "bisa", "akan", "atau", "lagi"
}

# =======================
# Load & Clean Data
# =======================
def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV tidak ditemukan: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if 'ulasan' not in df.columns or 'rating' not in df.columns:
        raise ValueError("CSV harus punya kolom 'ulasan' dan 'rating'")

    df['ulasan'] = df['ulasan'].fillna('').astype(str)

    def label_sentiment(r):
        if r >= 4:
            return 'positive'
        elif r == 3:
            return 'neutral'
        else:
            return 'negative'

    df['sentiment'] = df['rating'].apply(label_sentiment).str.lower().str.strip()
    return df

DF = load_data()

# =======================
# Routes
# =======================
@app.route('/')
def dashboard():
    return render_template("dashboard.html")

@app.route('/api/reviews')
def api_reviews():
    data = DF[['ulasan', 'rating', 'sentiment']].to_dict(orient='records')
    return jsonify({
        "recordsTotal": len(data),
        "recordsFiltered": len(data),
        "data": data
    })

@app.route('/api/sample_examples')
def api_examples():
    out = {}
    for s in ['positive', 'negative', 'neutral']:
        arr = DF[DF['sentiment'] == s]['ulasan'].dropna().head(5).tolist()
        if not arr:
            arr = DF['ulasan'].dropna().head(5).tolist()
        out[s] = arr
    return jsonify(out)

@app.route('/api/top_words')
def api_top_words():
    sentiment = request.args.get('sentiment', None)
    topn = int(request.args.get('topn', 30))

    print(f"Sentiment parameter received: {sentiment}")  # Debug log

    df_filtered = DF if not sentiment else DF[DF['sentiment'] == sentiment]

    words = []
    for text in df_filtered['ulasan']:
        for w in re.findall(r'\w+', text.lower()):
            if w not in STOPWORDS:
                words.append(w)

    counts = Counter(words).most_common(topn)

    # Format langsung untuk wordcloud2.js: [["kata", jumlah], ...]
    data = [[w, c] for w, c in counts]

    print("Top words data:", data)  # Debug log
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
