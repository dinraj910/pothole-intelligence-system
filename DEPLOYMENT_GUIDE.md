# 🚀 Free Deployment Guide

This guide details how to deploy the **Road Damage AI** web application completely for free. 

Running deep learning pipelines (like PyTorch and our 3-model YOLOv8 ensemble) requires significant memory. Standard free tiers (like Render, Heroku, or Fly.io) limit RAM to roughly 256MB–512MB, which often causes PyTorch to crash with "Out Of Memory" (OOM) errors.

To solve this, we provide two methods:
1. **Hugging Face Spaces (Recommended 🌟)**: Huge RAM allowance (16GB) completely for free.
2. **Render (Web Service)**: Uses our pre-configured `render.yaml`, but pushes the limits of their 512MB free tier.

---

## Method 1: Hugging Face Spaces (Highly Recommended)

Hugging Face provides free Docker hosting with **16GB of RAM** and **2 vCPUs**, making it the absolute best free option for Python AI web applications.

### Step 1: Create a Hugging Face Space
1. Create a free account at [Hugging Face](https://huggingface.co/).
2. Click your profile picture (top right) -> **New Space**.
3. Fill out the details:
   - **Space name**: `road-damage-ai` (or similar)
   - **License**: `MIT`
   - **Select the Space SDK**: Choose **Docker** (Blank).
   - **Space Hardware**: Free tier (2vCPU · 16GB)
4. Click **Create Space**.

### Step 2: Add a Dockerfile to the Repository
Hugging Face needs a `Dockerfile` to run our Flask app. Add a file named `Dockerfile` to the root of this project with the following content:

```dockerfile
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

# Run the app 
CMD ["gunicorn", "--workers", "1", "--threads", "2", "--timeout", "120", "--bind", "0.0.0.0:7860", "app.app:app"]
```

### Step 3: Connect directly from GitHub (GitHub Sync)

Since your project is already hosted on GitHub, you can connect it directly to Hugging Face without running any terminal commands!

1. Push your newly created `Dockerfile` to your GitHub repository.
2. Go to your newly created Hugging Face Space.
3. Click on the **Settings** tab.
4. Scroll down to the **GitHub repo sync** section.
5. Provide your GitHub Repository URL (e.g., `https://github.com/<your-username>/Pothole-AI-System`).
6. Click **Set up GitHub sync**.
7. Hugging Face will automatically pull the code from your GitHub repository, build the Docker container using your `Dockerfile`, and deploy the application. Every time you push updates to GitHub, it will automatically deploy the newly updated version to your Space!

**Alternative (Terminal Push):** 
If you ever want to push from your local terminal instead of linking GitHub, you can add Space as a remote:
```bash
git remote add space https://huggingface.co/spaces/<your-username>/road-damage-ai
git push space main -f
```

---

## Method 2: Render.com (Pre-configured)

We have already configured a `render.yaml` and `Procfile` for Render. The `Procfile` is specifically tuned with `gunicorn --workers 1` to squeeze PyTorch into Render's 512MB RAM free limit.

*Note: Render's free tier spins down after 15 minutes of inactivity. The next time someone visits, the app will experience a "Cold Start" taking about 30–60 seconds as the 3 models load into memory.*

### Step 1: Push Project to GitHub
1. Ensure your entire project (including the YOLOv8 weights) is pushed to a public or private repository on GitHub.

### Step 2: Deploy to Render
1. Create a free account at [Render](https://render.com/).
2. Click **New +** -> **Blueprint** (or **Web Service**).
3. Connect your GitHub account and select your repository.
4. Render will automatically detect the `render.yaml` file in the repository.
5. Click **Apply**. Render will automatically provision the environment, install the `requirements.txt`, and launch your app using the `Procfile`.

### Configuration Notes for Render
If you provision manually (without the blueprint), ensure you use these settings:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --workers 1 --timeout 120 --chdir app app:app` (This is already in our `Procfile`).

---

## 🛠️ Model Weights Management (Git LFS)

By default, GitHub restricts file uploads strictly to 100MB. Our 3 models (`model_a_rdd2022.pt`, etc.) are around 22MB each (66MB total), which easily fits inside standard Git without issue!

However, if you train larger models in the future, you will need to use [Git Large File Storage (Git LFS)](https://git-lfs.com/) to push them to GitHub or Hugging Face.

To setup Git LFS (if needed):
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add app/RoadDamageAI_Phase1/weights/*.pt
git commit -m "Track model weights with LFS"
```
