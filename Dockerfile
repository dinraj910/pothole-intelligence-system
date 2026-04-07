# Use a lightweight Python base image
FROM python:3.11-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face default)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirement list and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the 'app' directory contents directly to use as the application root
# This puts app.py, utils/, and RoadDamageAI_Phase1/ in the WORKDIR
COPY --chown=user app/ .

# Hugging Face routes traffic to port 7860 by default
ENV PORT=7860
EXPOSE 7860

# Run the app 
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:7860", "app:app"]