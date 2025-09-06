FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "automl_pipeline.pipeline", "--config", "examples/sample_config.yaml"]