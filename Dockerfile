FROM python:latest

COPY . /app
WORKDIR /app

RUN pip install .

CMD ["python", "-m", "diffusion.train"]