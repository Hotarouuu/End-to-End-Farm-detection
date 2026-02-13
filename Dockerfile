FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
