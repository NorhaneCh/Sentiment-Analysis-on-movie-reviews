import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Define the max length
max_length = 2493

# Load the JSON string from the file
with open('tokenizer.json', 'r') as file:
    tokenizer_json = file.read()

# Recreate the tokenizer from the JSON string
tokenizer = tokenizer_from_json(tokenizer_json)

# Define a function to preprocess the input review


def preprocess_review(review, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(
        sequences, maxlen=max_length, padding='post')
    return padded_sequence


# Define a function to predict the sentiment
def predict_sentiment(review, model, tokenizer, max_length):
    padded_sequence = preprocess_review(review, tokenizer, max_length)
    prediction = model.predict(padded_sequence)
    st.write(f"Prediction value : {prediction[0][0]}")
    sentiment = 'positive' if prediction >= 0.5 else 'negative'
    return sentiment


st.markdown(
    """
    <style>
    .center-text {
        text-align: center;
    }
    </style>
    <div class="center-text">
        <h1>Movie Reviews</h1>
    </div>
    """,
    unsafe_allow_html=True
)
image = Image.open('movies.jpg')
st.image(image)
review = st.text_input("Enter Movie Review")

if st.button('Predict'):
    if review:
        model = load_model('model.h5')
        sentiment = predict_sentiment(review, model, tokenizer, max_length)
        st.write(f"Predicted Sentiment : {sentiment}")
    else:
        st.write("Please enter a movie review to predict its sentiment.")
