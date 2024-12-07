import nltk
import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import string
import re
import streamlit as st
from nltk.corpus import stopwords
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def load_stop_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stop_words = set(file.read().splitlines())
    return stop_words

def load_bert_model(model_path):
    model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-cased', num_labels=9)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def load_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
    return tokenizer

# GPT-2 LLM modelini yükleme
def load_gpt2_model():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return model, tokenizer

# Bert model path
bert_path = "BertModel.pth"
bert_model = load_bert_model(bert_path)
stop_words_path = "tr_stop_word.txt"
stop_words = load_stop_words(stop_words_path)
tokenizer = load_tokenizer()

# GPT-2 Modeli
gpt2_model, gpt2_tokenizer = load_gpt2_model()

categories = {
    0: 'Magazin',
    1: 'Siyaset',
    2: 'Sağlık',
    3: 'Spor',
    4: 'Kültür-Sanat',
    5: 'Turizm',
    6: 'Finans-Ekonomi',
    7: 'Bilim-Teknoloji',
    8: 'Çevre'
}

# Metni Temizleme
def clean_text(text):
    text = text.replace('İ', 'i')
    text = text.replace('ı', 'i')
    text = text.replace('İ', 'i')
    text = text.replace('Ç', 'c')
    text = text.replace('ç', 'c')
    text = text.replace('Ğ', 'g')
    text = text.replace('ğ', 'g')
    text = text.replace('Ö', 'o')
    text = text.replace('ö', 'o')
    text = text.replace('Ş', 's')
    text = text.replace('ş', 's')
    text = text.replace('Ü', 'u')
    text = text.replace('ü', 'u')
    text = text.lower()
    text = re.sub(r'\d+•', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# Durak kelimelerini kaldırma
def remove_stopwords(text):
    words = text.split()
    clean_words = [word for word in words if word not in stop_words]
    return ' '.join(clean_words)

# Metin ön işleme
def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return text

# Streamlit UI
st.title("Haber Sınıflandırma ve Sonuç Üretme Uygulaması")
st.write("Lütfen en az 100 kelimelik bir haber metni giriniz:")

input_text = st.text_area("Haber Metni", height=200)

if st.button("Sınıflandır"):
    if not input_text.strip():  # Eğer metin boşsa
        st.warning("Lütfen bir haber metni giriniz.")
    else:
        words = input_text.split()
        if len(words) < 100:
            st.warning("Lütfen en az 100 kelime giriniz!")
        else:
            processed_text = preprocess_text(input_text)
            processed_text = processed_text[:512]  # BERT'in girdi uzunluğu sınırlıdır

            # Model ile tahmin
            with torch.no_grad():
                inputs = tokenizer(processed_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                outputs = bert_model(input_ids, attention_mask=attention_mask)

            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            bert_category = categories[predicted_class]

            st.success(f"BERT modeline göre haber '{bert_category}' kategorisine aittir.")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            tokenizer = AutoTokenizer.from_pretrained("mukayese/transformer-turkish-summarization")
            model = AutoModelForSeq2SeqLM.from_pretrained("mukayese/transformer-turkish-summarization")

            inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
            outputs = model.generate(**inputs)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.success(f"Mukayese Conclusion: {summary}")

