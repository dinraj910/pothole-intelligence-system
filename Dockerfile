# Use a lightweight Python base image
FROM python:3.11-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Hugging Face routes traffic to port 7860 by default
ENV PORT=7860
EXPOSE 7860

# Run the app — --chdir app puts app/ on Python path so 'utils' module is found
CMD ["gunicorn", "--chdir", "app", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:7860", "app:app"]