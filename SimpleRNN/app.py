import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb  # Preloaded dataset in keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from pathlib import Path
import streamlit as st

# ----------------------------
# Load IMDB word index helpers
# ----------------------------
word_index = imdb.get_word_index()  # word -> id (without +3 offset)
rev_keyval = {value: key for key, value in word_index.items()}  # id -> word

# ----------------------------
# Load model (robust path)
# ----------------------------
HERE = Path(__file__).parent                     # folder containing this app.py
MODEL_PATH = HERE / "simple_rnn_imdb.h5"         # model file sits next to app.py
if not MODEL_PATH.exists():
    # also try repo-root path in case you moved files
    ALT_PATH = HERE.parent / "SimpleRNN" / "simple_rnn_imdb.h5"
    if ALT_PATH.exists():
        MODEL_PATH = ALT_PATH

# cache the loaded model so Streamlit doesn't reload every run
@st.cache_resource
def get_model():
    assert MODEL_PATH.exists(), f"Model file not found at: {MODEL_PATH}"
    return load_model(str(MODEL_PATH))

model = get_model()

# ----------------------------
# Utilities
# ----------------------------
def decode_review(encoded_review):
    # Convert token IDs back to words (IMDB uses +3 offset for real words)
    return ' '.join(rev_keyval.get(i - 3, '?') for i in encoded_review)

def preprocess_text(text):
    # Lowercase + split
    words = text.lower().split()
    # Map words to ids; unknown -> 2, then +3 to match IMDB encoding
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    # Pad/truncate to length 500
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
