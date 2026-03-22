FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

RUN pip install --no-cache-dir earth2studio fastapi uvicorn pydantic numpy runpod

COPY src/ /app/src/
COPY test_input.json /app/test_input.json

ENV EARTH2STUDIO_CACHE=/models

VOLUME /models

CMD ["python", "-u", "-m", "src.handler"]
