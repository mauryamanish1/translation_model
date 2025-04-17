import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

# --- Constants ---
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 30
PKL_FILE = "paragraphs_with_embeddings_v2.pkl"
SIMILARITY_THRESHOLD = 0.4


# # --- UI Config ---
# st.title("🔍 Multilingual Paragraph Search")
# st.subheader("Built for PDF content in 2-column layout")

# query = st.text_input("Enter a detailed query (minimum 5 words):")

# # 👇 Add this slider to let user choose similarity threshold
# threshold = st.slider(
#     "Similarity Threshold (lower = more results, higher = more relevant)", 
#     min_value=0.1, 
#     max_value=0.9, 
#     value=0.4, 
#     step=0.05
# )



# --- Init ---
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_dataset():
    return pd.read_pickle(PKL_FILE)

model = load_model()
translator = Translator()
df = load_dataset()

# --- App UI ---
st.title("📄 Multilingual Paragraph Search")
st.markdown("Search through multilingual technical documents.")

query = st.text_input("Enter search query (at least 6 words):")
# translate_toggle = st.checkbox("🔁 Include translated paragraph (optional)", value=False)

if st.button("🔍 Search"):

    if not query or len(query.strip().split()) < 6:
        st.warning("Please enter a search query with more than 5 words.")
    else:
        query_embedding = model.encode(query)
        para_embeddings = np.vstack(df['embedding'].to_numpy())
        scores = cosine_similarity([query_embedding], para_embeddings)[0]

        df['score'] = scores
        df_filtered = df[df['score'] >= SIMILARITY_THRESHOLD]
        df_top = df_filtered.sort_values(by="score", ascending=False).head(TOP_K).copy()

        if df_top.empty:
            st.info("No matching paragraphs found.")
        else:
            st.success(f"Found {len(df_top)} matching paragraphs.")
            
            # Optional Translation
            # if translate_toggle:
            #     df_top["translated_paragraph"] = df_top["paragraph"].apply(
            #         lambda x: translator.translate(x, dest="en").text
            #     )

            # Display table
            display_cols = ["doc_name", "page_number", "paragraph", "language", "score"]
            # if translate_toggle:
            #     display_cols.append("translated_paragraph")
            st.dataframe(df_top[display_cols], use_container_width=True)

            # Download
            def convert_df_to_csv(df_export):
                return df_export.to_csv(index=False, encoding="utf-8-sig")

            csv = convert_df_to_csv(df_top[display_cols])
            st.download_button(
                label="📥 Download results as CSV",
                data=csv,
                file_name="filtered_paragraphs.csv",
                mime="text/csv"
            )