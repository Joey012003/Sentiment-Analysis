import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('tokenizer/')
    model = BertForSequenceClassification.from_pretrained('model/')
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("Emotion Detection from Tweets")
st.write("Enter a tweet or short text to detect its emotion.")

user_input = st.text_area("Your text here")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        emotion_id = torch.argmax(probs, dim=1).item()
        emotions = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust']
        st.success(f"**Detected Emotion:** {emotions[emotion_id]} ({probs.max().item()*100:.2f}% confidence)")
