FROM python:3.11

WORKDIR /app

RUN mkdir -p /app/logs

COPY . /app

RUN pip install .
