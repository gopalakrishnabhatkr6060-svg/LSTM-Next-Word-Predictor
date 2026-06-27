import streamlit as st
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="LSTM Next Word Predictor",
    page_icon="📝",
    layout="centered"
)

st.title("📝 LSTM Next Word Prediction")
st.write("Enter a sentence and let the LSTM model predict the next words.")

# ----------------------------
# Load Model & Artifacts
# ----------------------------
@st.cache_resource
def load_lstm_model():
    try:
        return load_model("model.h5")
    except Exception as e:
        st.error(f"Failed to load model.h5: {e}")
        st.stop()

@st.cache_resource
def load_tokenizer():
    try:
        with open("tokenizer.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load tokenizer.pkl: {e}")
        st.stop()

@st.cache_resource
def load_max_len():
    try:
        with open("max_len.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load max_len.pkl: {e}")
        st.stop()

model = load_lstm_model()
tokenizer = load_tokenizer()
max_len = int(load_max_len())  # Ensure it's a Python int

# Build reverse mapping once for fast lookup
reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}

# ----------------------------
# Predict Next Word
# ----------------------------
def predict_next_word(model, tokenizer, text, max_len, reverse_word_index):
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Handle unknown words / empty input
    if not token_list:
        return ""

    token_list = pad_sequences(
        [token_list],
        maxlen=max_len - 1,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)

    # Get the index of the highest probability word
    predicted_index = int(np.argmax(predicted, axis=-1)[0])

    # Return the word using reverse mapping
    return reverse_word_index.get(predicted_index, "")

# ----------------------------
# Generate Text
# ----------------------------
def generate_text(model, tokenizer, seed_text, n_words, max_len, reverse_word_index):
    current_text = seed_text.strip()

    for _ in range(n_words):
        next_word = predict_next_word(
            model,
            tokenizer,
            current_text,
            max_len,
            reverse_word_index
        )

        if next_word == "":
            break

        current_text += " " + next_word

    return current_text

# ----------------------------
# UI
# ----------------------------
seed = st.text_area(
    "Enter starting sentence",
    placeholder="Example: once upon a time",
    height=100
)

num_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=20,
    value=5
)

if st.button("🚀 Generate"):
    if seed.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating..."):
            generated = generate_text(
                model,
                tokenizer,
                seed,
                num_words,
                max_len,
                reverse_word_index
            )

        st.success("Generated Text")
        st.text_area("Output", value=generated, height=150, label_visibility="collapsed")
