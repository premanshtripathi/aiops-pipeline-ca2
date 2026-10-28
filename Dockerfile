# 1. Use official lightweight Python slim image
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set working directory inside the container
WORKDIR /app

# 4. Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# 5. Install dependencies without caching wheel files to keep image size small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy application code, test suite, and saved model artifact
COPY main.py ./
COPY models/ ./models/
COPY tests/ ./tests/

# 7. Expose port 8000 for FastAPI
EXPOSE 8000

# 8. Start Uvicorn server bound to 0.0.0.0 so it is accessible outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]