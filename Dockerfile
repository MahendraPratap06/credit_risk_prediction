FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update -y && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 1080

CMD ["python", "application.py"]
