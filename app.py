import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from deep_translator import GoogleTranslator
from langdetect import detect

# Download necessary NLTK data (run once)
nltk.download('stopwords')

# --- Page Configuration ---
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="centered")

# --- Custom CSS for Styling (DÜZELTİLEN KISIM) ---
page_bg_img = """
<style>
/* 1. Arka Planı Ayarla (Koyu Gradient) */
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(to right top, #051937, #004d7a, #008793, #00bf72, #a8eb12);
}

/* 2. Başlık ve Yazı Renklerini Beyaz Yap */
h1 { color: white !important; text-shadow: 2px 2px 4px #000000; }
p, .stMarkdown, label { color: white !important; }

/* 3. Metin Kutusunu (Text Area) Düzelt */
/* Kutunun içi beyaz, yazı siyah olsun */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* 4. Butonları Güzelleştir */
.stButton>button {
    color: #051937 !important;
    background-color: #a8eb12 !important;
    font-weight: bold !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- 1. Load Model and Vectorizer ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_assets()

# --- 2. Text Preprocessing Function ---
def preprocess_text(text):
    text = text.lower()
    
    # Expand contractions
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"cannot", "can not", text)
    
    # Remove HTML and Special Chars
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    words = text.split()
    
    # Stopwords (Keep negations)
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none'}
    stop_words = stop_words - negation_words
    
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    
    return ' '.join(words)

# --- 3. Translation Function ---
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang == 'en':
            return text
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

# --- 4. Rule-Based Correction ---
def rule_based_correction(text, prediction, confidence):
    text_lower = text.lower()
    
    strong_positive_words = ["awesome", "amazing", "excellent", "fantastic", "perfect", "masterpiece", "brilliant", "wonderful", "superb", "spectacular"]
    negation_positive_patterns = ["not bad", "not too bad", "not terrible", "not awful", "not boring", "not worst", "watchable", "not a waste"]
    strong_negative_words = ["awful", "terrible", "disgusting", "trash", "garbage", "worst", "waste of time", "unwatchable", "horrible"]

    for pat in negation_positive_patterns:
        if pat in text_lower: return 'positive', 0.92

    for word in strong_positive_words:
        if word in text_lower and f"not {word}" not in text_lower: return 'positive', 0.98

    for word in strong_negative_words:
        if word in text_lower and f"not {word}" not in text_lower: return 'negative', 0.98

    return prediction, confidence

# --- Main Interface ---
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter your review below (in English or any other language). The AI will analyze the sentiment.")

if st.button("🔄 Clear Cache"):
    st.cache_resource.clear()
    st.rerun()

if model is None or vectorizer is None:
    st.error("ERROR: Model files not found.")
else:
    user_input = st.text_area("Your Review:", height=150, placeholder="Type your movie review here...")

    if st.button("Analyze Sentiment 🚀"):
        if not user_input.strip():
            st.warning("Please enter a review.")
        else:
            with st.spinner("Analyzing..."):
                english_text = translate_to_english(user_input)
                
                if user_input.strip().lower() != english_text.strip().lower():
                    st.info(f"🇬🇧 Translated to English: {english_text}")

                cleaned_text = preprocess_text(english_text)
                vectorized_text = vectorizer.transform([cleaned_text])
                
                prediction_raw = model.predict(vectorized_text)[0]
                
                try:
                    proba = model.predict_proba(vectorized_text)[0]
                    confidence_raw = proba[1] if prediction_raw in ['positive', 1] else proba[0]
                except:
                    confidence_raw = 0.85

                final_pred, final_conf = rule_based_correction(english_text, prediction_raw, confidence_raw)
                
                st.divider()
                col1, col2 = st.columns([1, 4])
                
                if final_pred == 'positive' or final_pred == 1:
                    with col1: st.markdown("# 😊")
                    with col2:
                        st.success(f"**Result: POSITIVE**")
                        st.caption(f"Confidence: %{final_conf*100:.1f}")
                else:
                    with col1: st.markdown("# 😞")
                    with col2:
                        st.error(f"**Result: NEGATIVE**")
                        st.caption(f"Confidence: %{final_conf*100:.1f}")