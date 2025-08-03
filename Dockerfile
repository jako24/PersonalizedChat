FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . /app/

# Make OpenAI API key available as a build secret
# --mount=type=secret,id=openai_api_key,dst=/run/secrets/openai_api_key
# This will mount the secret file into the container at the specified destination.
# The script build_index.py needs to be modified to read the key from this file.
ARG OPENAI_API_KEY
RUN --mount=type=secret,id=openai_api_key,dst=/run/secrets/openai_api_key python scripts/build_index.py

EXPOSE 8000
ENV PYTHONPATH=/app
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
