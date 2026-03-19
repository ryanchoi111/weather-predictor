FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn pydantic numpy runpod

COPY src/ /app/src/

ENV EARTH2STUDIO_CACHE=/models

VOLUME /models

CMD ["python", "-u", "src/handler.py"]
