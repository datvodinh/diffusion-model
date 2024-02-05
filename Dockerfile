FROM python:latest

COPY . /app
WORKDIR /app

RUN pip install .

CMD ["python", "app.py"]