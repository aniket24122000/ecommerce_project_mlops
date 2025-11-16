# Use lightweight Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy only FastAPI app + models
COPY app/ app/
COPY models/ models/

# Install dependencies
RUN pip install fastapi uvicorn joblib numpy pandas

# Expose FastAPI port
EXPOSE 8000

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
