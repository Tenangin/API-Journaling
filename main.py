from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import json
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.utils import custom_object_scope
from utils.attention_layer import AttentionLayer  
import pickle

from utils.text_utils import (
    build_slang_dictionary,
    build_stopwords,
    predict_sentiment_per_sentence
)

model = tf.keras.models.load_model('model/best_model.h5',
                                   custom_objects={'AttentionLayer': AttentionLayer})

with open('model/tokenizer.json') as f:
    tokenizer = tokenizer_from_json(json.load(f))
with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
max_len = 20

slangwords = build_slang_dictionary()
stopwords = build_stopwords()

# === API ===
app = FastAPI()

class SentimentRequest(BaseModel):
    userId: str
    content: str 

def verify_token(authorization: Optional[str]):
    # if not authorization or not authorization.startswith("Bearer "):
    #     raise HTTPException(status_code=401, detail="Invalid or missing token")
    # token = authorization.split(" ")[1]
    # if token != "dummy-token":  
    #     raise HTTPException(status_code=403, detail="Unauthorized")
    pass
@app.post("/analyze")
async def analyze_sentiment(request: SentimentRequest, authorization: Optional[str] = Header(None)):
    verify_token(authorization)

    results = predict_sentiment_per_sentence(
        request.content, model, tokenizer, label_encoder, max_len, slangwords, stopwords
    )

    payload = {
        "content": request.content,
        "sentiment": results
    }
    headers = {
        "Authorization": "Bearer dummy-token",
        "Content-Type": "application/json"
    }
    # response = requests.post("https://dummy.api/simpan", headers=headers, json=payload)

    return {
        "userId": request.userId,
        "results": results,
        # "api_response": response.status_code
    }
