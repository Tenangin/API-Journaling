
API ini merupakan backend service berbasis **FastAPI** untuk melakukan **analisis sentimen teks berbahasa Indonesia**. Model yang digunakan adalah **Bidirectional LSTM + Attention Layer** yang dilatih menggunakan dataset opini publik berbahasa Indonesia.

---

## Fitur Utama

- Preprocessing teks: normalisasi kata gaul, hapus stopword, dsb.
- Prediksi sentimen per kalimat.
- Arsitektur: BiLSTM + Custom Attention Layer (TensorFlow).
- Label Sentimen: `Anger`, `Fear`, `Joy`, `Love`, `Neutral`, `Sad`.
- Format respons JSON, siap digunakan frontend.

---

## Menjalankan API
### 1. Clone dan Install
```
git clone https://github.com/Tenangin/Tenangin-Analisis-Sentimen-.git
cd Tenangin-Analisis-Sentimen-
pip install -r requirements.txt
```
### 2. Jalankan Server FastAPI
uvicorn main:app --reload
Server akan tersedia di: http://127.0.0.1:8000/analyze

Endpoint
POST /analyze
Melakukan analisis sentimen terhadap input teks.
Request Body
```
{
  "userId": "string",
  "content": "hari ini aku sangat senang dan bersyukur"
}
```
Response
```
{
  "userId": "string",
  "results": [
    "Anger", "Sad", "Joy", "Neutral", "Love", "Fear"
  ]
}
```
