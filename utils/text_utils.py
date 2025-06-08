import re, string, json, requests
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_slang_word_list(url):
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    slang_dict = {}
    for line in lines:
        if '=' in line:
            key, val = line.split('=', 1)
            slang_dict[key.strip()] = val.strip()
    return slang_dict

def get_stopwords(urls):
    all_stopwords = set()
    for url in urls:
        response = requests.get(url)
        words = response.text.strip().split('\n')
        if words and 'stopword' in words[0].lower():
            words.pop(0)
        all_stopwords.update(w.strip().lower() for w in words)
    return all_stopwords

def build_slang_dictionary():
    slang_url = 'https://raw.githubusercontent.com/King-srt/Indonesia-Slang-Dictionary/refs/heads/main/dictionary_indonesia.txt'
    slangwords = get_slang_word_list(slang_url)
    with open('utils/custom_slang.json', 'r', encoding='utf-8') as f:
        custom_slang = json.load(f)
    slangwords.update(custom_slang)
    return slangwords

def build_stopwords():
    urls = [
        'https://raw.githubusercontent.com/yasirutomo/python-sentianalysis-id/master/data/feature_list/stopwordsID.txt',
        'https://raw.githubusercontent.com/Braincore-id/IndoTWEEST/main/stopwords_twitter.csv'
    ]
    stop_words = get_stopwords(urls)
    with open('utils/add_stopwords.json', 'r', encoding='utf-8') as f:
        add_stopwords = json.load(f)
    stop_words.update(add_stopwords)
    return stop_words

def processing_text_id(text, slangwords, stop_words):
    if pd.isnull(text):
        return []
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = word_tokenize(text)
    tokens = [slangwords.get(token, token) for token in tokens]
    tokens = [token for token in tokens if token not in stop_words and token.strip()]
    return tokens

def predict_sentiment_per_sentence(text, model, tokenizer, label_encoder, max_len, slangwords, stop_words):
    sentences = sent_tokenize(text)
    print(sentences)
    results = []
    for sentence in sentences:
        cleaned_tokens = processing_text_id(sentence, slangwords, stop_words)
        processed_text = ' '.join(cleaned_tokens)
        if not processed_text.strip():
            results.append("unknown")
            continue
        print(sentence)
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
        prediction = model.predict(padded_sequence, verbose=0)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
        results.append(predicted_label)
    return results
