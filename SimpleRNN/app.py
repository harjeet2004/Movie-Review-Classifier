import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb  ## Preloaded dataset in keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

## Load the IMDB dataset with word indices 
word_index=imdb.get_word_index() ## Dictionary with the words as keys and the corresponding one-hot encoded indices as values
rev_keyval={value:key for key,value in word_index.items()}

model=load_model('SimpleRNN/simple_rnn_imdb.h5')
# Function to decode reviews (numbers → words)
def decode_review(encoded_review):
    # IMDB dataset quirk:
    # - In the actual training data, words are stored with an offset of +3
    #   because 0,1,2 are reserved (0=padding, 1=start, 2=unknown).
    # - Example: word_index says "good"=3, but in the encoded review it's stored as 6.
    #
    # So when decoding back to words, we subtract 3 to undo that offset
    # and look up the real word in the reverse dictionary (rev_keyval).
    # If the word isn't found, we put '?' as a placeholder.
    return ' '.join(rev_keyval.get(i-3, '?') for i in encoded_review)


# Function to preprocess user input (words → numbers)
def preprocess_text(text):
    # Step 1: Make the text lowercase and split into individual words
    words = text.lower().split()
    
    # Step 2: Convert each word into its numeric ID
    # - word_index gives the base ID (e.g., "good"=3).
    # - If the word isn't found, default to 2 (UNKNOWN token).
    # - Then add +3 to align with IMDB's stored format
    #   (since 0,1,2 are reserved).
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    
    # Step 3: Pad or truncate the list of numbers to a fixed length (500 tokens)
    # - Neural networks need inputs of the same size
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    
    # Step 4: Return the padded numeric sequence (ready for model prediction)
    return padded_review


## design the streamlit app

'''
- st.title is Streamlit’s built-in heading style → it wants to keep things
  neat and readable, so if the text is long, it automatically breaks it
  into multiple lines to fit the page.

- st.markdown is more like writing raw text/HTML → it gives you more control,
  so the text will stay in a single line as long as it fits on the screen.

- In simple terms:
  • st.title = “Streamlit decides formatting for you (auto-wrap).”
  • st.markdown = “You decide formatting (no forced wrapping).”

'''

st.markdown("## Movie Review Classifier (Sentiment Analysis)")

st.write('Enter a movie review to classify it as postive or negative')

user_input=st.text_area('Movie Review')

if st.button('Classify'):

    prepro_inp=preprocess_text(user_input)

    ## Make prediction

    prediction=model.predict(prepro_inp)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review')



