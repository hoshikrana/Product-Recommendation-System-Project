from flask import Flask, request, jsonify, render_template
from surprise.dump import load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gc
import os

app = Flask(__name__)

# Globals for lazy loading
best_model = None
df = None
tfidf_vectorizer = None
tfidf_matrix_content = None

model_filename = 'best_model.pkl'
data_path = 'dataset.csv'

# === Load CF model lazily ===
def load_model(filename):
    global best_model
    if best_model is None:
        try:
            _, best_model = load(filename)
            print(" Model loaded successfully")
        except Exception as e:
            print(" Error loading model:", e)

# === Load data and initialize TF-IDF lazily ===
def initialize_data_and_tfidf(path):
    global df, tfidf_vectorizer, tfidf_matrix_content
    if df is None:
        try:
            df = pd.read_csv(path).sample(n=10000, random_state=42)  # Trimmed
            print(f" Data loaded: {df.shape}")

            df['combined_text'] = (
                df['product_category_name_english'].fillna('') +
                (' ' + df['Brand'].fillna('')) * 3 +
                (' ' + df['Name'].fillna('')) * 3 +
                ' ' + df['Description'].fillna('') +
                ' ' + df['Tags'].fillna('')
            )

            tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf_matrix_content = tfidf_vectorizer.fit_transform(df['combined_text'])
            print(f" TF-IDF initialized: {tfidf_matrix_content.shape}")
            gc.collect()
        except Exception as e:
            print(f" TF-IDF Init Error: {e}")

# === Content-based recommendation ===
def Content_Base_Recomendation(dataframe, search_term, top_n=10):
    try:
        search_vector = tfidf_vectorizer.transform([search_term])
        cos_sim = cosine_similarity(search_vector, tfidf_matrix_content)
        similar_items = sorted(enumerate(cos_sim[0]), key=lambda x: x[1], reverse=True)
        recommended_indexes = [x[0] for x in similar_items if x[1] > 0.0][:top_n]

        if not recommended_indexes:
            return pd.DataFrame()

        return dataframe.iloc[recommended_indexes][['Name', 'Brand', 'ReviewCount', 'review_score']]
    except Exception as e:
        print(f" CB Error: {e}")
        return pd.DataFrame()

# === Hybrid recommendation ===
def hybrid_recommendations(user_id, search_term, top_n=10):
    initialize_data_and_tfidf(data_path)
    load_model(model_filename)
    if df is None or best_model is None or tfidf_vectorizer is None:
        return pd.DataFrame()

    cb_recs = Content_Base_Recomendation(df, search_term, top_n=top_n * 2)
    if cb_recs.empty:
        print(" No CB recommendations")
        return pd.DataFrame()

    cb_item_ids = []
    for _, row in cb_recs.iterrows():
        match = df[(df['Name'] == row['Name']) & (df['Brand'] == row['Brand'])]
        if not match.empty:
            cb_item_ids.append(match.iloc[0]['product_id'])

    cf_predictions = [
        best_model.predict(user_id, iid) for iid in cb_item_ids if not pd.isna(iid)
    ]

    final_recs = []
    for pred in cf_predictions:
        if pred.iid in df['product_id'].values:
            details = df[df['product_id'] == pred.iid].iloc[0]
            final_recs.append({
                'Name': details['Name'],
                'Brand': details['Brand'],
                'ReviewCount': details['ReviewCount'],
                'Rating': pred.est
            })
    return pd.DataFrame(final_recs).sort_values(by='Rating', ascending=False).head(top_n)

# === Flask routes ===
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id_str = request.args.get('user_id')
    search_term = request.args.get('search_term', type=str)
    top_n = request.args.get('top_n', default=10, type=int)

    if not user_id_str or not search_term:
        return jsonify({"error": "Provide both 'user_id' and 'search_term'"}), 400

    try:
        user_id = float(user_id_str)
        recommendations = hybrid_recommendations(user_id, search_term, top_n)
        if recommendations.empty:
            return jsonify({"message": "No recommendations found."}), 404
        return jsonify(recommendations.to_dict(orient='records'))
    except Exception as e:
        print(f" API error: {e}")
        return jsonify({"error": "Something went wrong."}), 500



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render uses PORT env variable
    app.run(debug=True, host='0.0.0.0', port=port)