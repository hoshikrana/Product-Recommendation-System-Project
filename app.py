import streamlit as st
import pandas as pd
from surprise.dump import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Load CF model ===
model_filename = "best_model.pkl"
_, best_model = load(model_filename)

# === Load data ===
df = pd.read_csv("dataset.csv")

# === Initialize TF-IDF ===
if "combined_text" not in df.columns:
    df['combined_text'] = df['product_category_name_english'].fillna('') + \
                          (' ' + df['Brand'].fillna('')) * 3 + \
                          (' ' + df['Name'].fillna('')) * 3 + \
                          ' ' + df['Description'].fillna('') + \
                          ' ' + df['Tags'].fillna('')

tfidf_vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode')
tfidf_matrix_content = tfidf_vectorizer.fit_transform(df['combined_text'])

# === Recommendation Functions ===
def content_based_recommendation(search_term, top_n=10):
    search_vector = tfidf_vectorizer.transform([search_term])
    cos_sim = cosine_similarity(search_vector, tfidf_matrix_content)
    similar_items = sorted(list(enumerate(cos_sim[0])), key=lambda x: x[1], reverse=True)
    recommended_indexes = [x[0] for x in similar_items if x[1] > 0.0][:top_n]
    return df.iloc[recommended_indexes][['product_id', 'Name', 'Brand', 'ReviewCount', 'review_score']]

def hybrid_recommendations(user_id, search_term, top_n=10):
    cb_recs = content_based_recommendation(search_term, top_n * 2)
    if cb_recs.empty:
        return pd.DataFrame()

    cb_item_ids = []
    for _, row in cb_recs.iterrows():
        match = df[(df['Name'] == row['Name']) & (df['Brand'] == row['Brand'])]
        if not match.empty:
            cb_item_ids.append(match.iloc[0]['product_id'])

    cf_predictions = [best_model.predict(user_id, iid) for iid in cb_item_ids if not pd.isna(iid)]
    recommended_items = []
    for pred in cf_predictions:
        if pred.iid in df['product_id'].values:
            details = df[df['product_id'] == pred.iid].iloc[0]
            recommended_items.append({
                'Name': details['Name'],
                'Brand': details['Brand'],
                'ReviewCount': details['ReviewCount'],
                'Rating': pred.est
            })
    return pd.DataFrame(recommended_items).sort_values(by='Rating', ascending=False).head(top_n)

# === Streamlit UI ===
st.title("🛍️ Hybrid Product Recommendation System")
st.write("Enter a user ID and/or search term to get product recommendations.")

user_id = st.number_input("User ID", min_value=0.0, format="%.0f")
search_term = st.text_input("Search Term")
top_n = st.slider("Number of Recommendations", 5, 20, 10)

if st.button("Recommend"):
    if user_id and search_term:
        result = hybrid_recommendations(user_id, search_term, top_n)
    elif search_term:
        result = content_based_recommendation(search_term, top_n)
    else:
        st.warning("Please enter a search term (and optionally a user ID).")
        result = pd.DataFrame()

    if not result.empty:
        st.dataframe(result)
    else:
        st.info("No recommendations found.")