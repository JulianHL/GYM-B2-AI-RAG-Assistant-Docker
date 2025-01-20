FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}


EXPOSE 8501

# Comando por defecto para iniciar la aplicación
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
