import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb  # Preloaded dataset in keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from pathlib import Path
import streamlit as st
import os

# ----------------------------
# Load the IMDB dataset word index
# ----------------------------
word_index = imdb.get_word_index()  # word -> id (raw, without +3 offset)
rev_keyval = {value: key for key, value in word_index.items()}  # id -> word

# ----------------------------
# Robust model path resolution (works local & Streamlit Cloud)
# ----------------------------
HERE = Path(__file__).parent                                  # folder containing this app.py
CANDIDATES = [
    HERE / "simple_rnn_imdb.h5",                              # preferred: next to app.py
    HERE.parent / "simple_rnn_imdb.h5",                       # fallback: repo root
    HERE / "SimpleRNN" / "simple_rnn_imdb.h5",                # fallback: nested folder variant
]

MODEL_PATH = next((p for p in CANDIDATES if p.exists()), None)

@st.cache_resource
def get_model():
    assert MODEL_PATH is not None, (
        "Model file not found. Checked:\n" +
        "\n".join(f"- {p}" for p in CANDIDATES)
        + "\n\nIf you're deploying to Streamlit Cloud, ensure the file is committed "
          "to your repo (watch out for .gitignore with *.h5) and the path/casing matches."
    )
    return load_model(str(MODEL_PATH))

# Optional debug UI to inspect paths (safe to keep; prints only on demand)
with st.expander("🔧 Debug (paths & environment)"):
    st.write("Working directory:", os.getcwd())
    st.write("app.py directory:", str(HERE))
    st.write("Model candidates:", [str(p) for p in CANDIDATES])
    st.write("Resolved MODEL_PATH:", str(MODEL_PATH) if MODEL_PATH else None)
    if MODEL_PATH:
        st.write("Model exists:", MODEL_PATH.exists())

model = get_model()

# ----------------------------
# Utilities
# ----------------------------
def decode_review(encoded_review):
    # Convert token IDs back to words (IMDB uses +3 offset for real words)
    return ' '.join(rev_keyval.get(i - 3, '?') for i in encoded_review)

def preprocess_text(text):
    # 1) lowercase + split
    words = text.lower().split()
    # 2) map words -> ids; unknown -> 2, then +3 to match IMDB encoding (0,1,2 reserved)
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # 3) pad/truncate to fixed length the model expects
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# ----------------------------
# Streamlit UI
# ----------------------------
st.markdown("## Movie Review Classifier (Sentiment Analysis)")
st.write('Enter a movie review to classify it as positive or negative')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if not user_input.strip():
        st.warning("Please enter a movie review.")
    else:
        prepro_inp = preprocess_text(user_input)
        prediction = model.predict(prepro_inp)
        score = float(prediction[0][0])
        sentiment = 'Positive' if score > 0.5 else 'Negative'
        st.write(f"Sentiment: **{sentiment}** | Prediction Score: **{score:.4f}**")
else:
    st.write('Please enter a movie review')
