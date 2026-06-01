FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CONFIG_PATH=configs/debug.yaml

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs

EXPOSE 8000

CMD ["uvicorn", "src.churn.api:app", "--host", "0.0.0.0", "--port", "8000"]