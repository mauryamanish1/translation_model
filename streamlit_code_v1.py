# %%
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

# --- Config ---
PKL_FILE = "paragraphs_with_embeddings_v2.pkl"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.4
TOP_K = 30
SHOW_TRANSLATIONS = False  # Set to True if you want 'translated_paragraph'

# --- Load model and data ---
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data():
    return pd.read_pickle(PKL_FILE)

model = load_model()
df = load_data()
translator = Translator()

# --- Search Function ---
def search_paragraphs(query):
    query_embedding = model.encode(query)
    para_embeddings = np.vstack(df['embedding'].to_numpy())
    scores = cosine_similarity([query_embedding], para_embeddings)[0]
    df['score'] = scores
    df_filtered = df[df['score'] >= SIMILARITY_THRESHOLD]
    df_top = df_filtered.sort_values(by="score", ascending=False).head(TOP_K).copy()
    return df_top

# --- Filter Best per Language ---
def filter_best_per_language(df_results, original_query):
    if df_results.empty:
        return df_results

    translated_query = translator.translate(original_query, dest='en').text
    translated_query_embedding = model.encode(translated_query)

    ref_page = df_results.iloc[0]['page_number']
    best_by_lang = {}

    for _, row in df_results.iterrows():
        para = row['paragraph']
        lang = row['language']
        if abs(row['page_number'] - ref_page) > 3:
            continue
        try:
            translated_para = translator.translate(para, dest='en').text
            translated_para_embedding = model.encode(translated_para)
            sim = cosine_similarity([translated_query_embedding], [translated_para_embedding])[0][0]
            if lang not in best_by_lang or sim > best_by_lang[lang]['similarity']:
                best_by_lang[lang] = {
                    "row": row,
                    "similarity": sim,
                    "translated": translated_para
                }
        except Exception as e:
            print(f"Translation failed for lang {lang}: {e}")
            continue

    filtered_rows = []
    for lang, entry in best_by_lang.items():
        row = entry["row"]
        if SHOW_TRANSLATIONS:
            row['translated_paragraph'] = entry['translated']
        filtered_rows.append(row)

    return pd.DataFrame(filtered_rows)

# --- Streamlit UI ---
st.title("🔍 Multilingual Paragraph Search")
st.subheader("Built for pdf content in 2 Column layout")

query = st.text_input("Enter a detailed query (minimum 5 words):")

if st.button("Search"):

    if len(query.split()) < 5:
        st.warning("Please enter a more descriptive query (at least 5 words).")
    else:
        with st.spinner("Searching..."):
            results = search_paragraphs(query)
            filtered = filter_best_per_language(results, query)

            if filtered.empty:
                st.error("No matching paragraphs found.")
            else:
                st.success(f"Top {len(filtered)} matches found!")

                # Display results
                st.dataframe(filtered.drop(columns=["embedding"]), use_container_width=True)

                # Download option
                csv = filtered.drop(columns=["embedding"]).to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="filtered_paragraphs.csv",
                    mime="text/csv"
                )



