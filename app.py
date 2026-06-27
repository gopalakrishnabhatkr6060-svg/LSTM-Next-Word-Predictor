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
# Load Model
# ----------------------------
@st.cache_resource
def load_lstm_model():
    return load_model("model.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_max_len():
    with open("max_len.pkl", "rb") as f:
        return pickle.load(f)

model = load_lstm_model()
tokenizer = load_tokenizer()
max_len = load_max_len()


# ----------------------------
# Predict Next Word
# ----------------------------
def predict_next_word(model, tokenizer, text, max_len):

    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_len - 1,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)

    predicted_index = np.argmax(predicted, axis=-1)[0]

    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break

    return output_word


# ----------------------------
# Generate Text
# ----------------------------
def generate_text(model, tokenizer, seed_text, n_words, max_len):

    for _ in range(n_words):

        next_word = predict_next_word(
            model,
            tokenizer,
            seed_text,
            max_len
        )

        if next_word == "":
            break

        seed_text += " " + next_word

    return seed_text


# ----------------------------
# UI
# ----------------------------
seed = st.text_area(
    "Enter starting sentence",
    placeholder="Example: once upon a time"
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
                max_len
            )

        st.success("Generated Text")

        st.text_area(
            "",
            value=generated,
            height=150
        )