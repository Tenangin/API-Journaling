# Gunakan Python 3.9 sebagai base image
FROM python:3.9

# Tambahkan user tanpa akses root
RUN useradd -m -u 1000 user
USER user

# Set PATH environment
ENV PATH="/home/user/.local/bin:$PATH"

# Tentukan working directory di dalam container
WORKDIR /app

# Salin file requirements.txt terlebih dahulu untuk optimasi caching layer
COPY --chown=user ./requirements.txt ./requirements.txt

# Install dependencies dari requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Salin semua file dari direktori lokal ke dalam container
COPY --chown=user . .

# Jalankan aplikasi menggunakan uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
