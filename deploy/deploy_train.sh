#!/bin/bash
set -e

# ==============================================================================
# MoneyPrinter: Deploy the 'Train' Job (Cloud Run GPU)
# 
# This script builds a PyTorch Docker container, pushes it to Google Artifact 
# Registry, and provisions a Cloud Run Job outfitted with a dedicated NVIDIA L4
# GPU. You can trigger this job manually via the GCP Console or CLI.
# ==============================================================================

PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
REPO_NAME="moneyprinter-repo"
IMAGE_NAME="moneyprinter-train"
JOB_NAME="moneyprinter-train-job"
SERVICE_ACCOUNT="moneyprinter-trainer@${PROJECT_ID}.iam.gserviceaccount.com"

echo "🚀 Deploying MoneyPrinter 'Train' Job to Google Cloud Run (with GPU)..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# 1. Ensure Artifact Registry Repository exists
if ! gcloud artifacts repositories describe $REPO_NAME --location=$REGION >/dev/null 2>&1; then
    echo "📦 Creating Artifact Registry Repository: $REPO_NAME..."
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="Docker repository for MoneyPrinter images"
fi

# Authenticate Docker to Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# 2. Build and Push the Docker Image using Cloud Build (using the special train Dockerfile)
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
echo "🔨 Building PyTorch Docker image using Cloud Build..."
gcloud builds submit --tag $IMAGE_URL --config /dev/null --pack builder=gcr.io/buildpacks/builder:v1,path=.

# Alternatively, since we have a custom Dockerfile with PyTorch base image:
gcloud builds submit --tag $IMAGE_URL -f Dockerfile.train .

# 3. Create or Update the Cloud Run Job with GPU Configuration
# Note: Cloud Run requires limits to match requests for CPU/Memory when requesting GPUs.
echo "☁️  Deploying Cloud Run Job: $JOB_NAME (Requesting NVIDIA L4 GPU)..."
gcloud run jobs deploy $JOB_NAME \
    --image $IMAGE_URL \
    --region $REGION \
    --tasks 1 \
    --max-retries 0 \
    --command "bash" \
    --args "-c,gsutil cp gs://moneyprinter-artifacts-${PROJECT_ID#moneyprinter-}/datasets/data.parquet data.parquet && python main.py package && python training_bundle/train.py --data data.parquet --output results && gsutil cp results/best_model.pth gs://moneyprinter-artifacts-${PROJECT_ID#moneyprinter-}/models/best_model.pth && gsutil cp results/scaler.pkl gs://moneyprinter-artifacts-${PROJECT_ID#moneyprinter-}/models/scaler.pkl" \
    --service-account $SERVICE_ACCOUNT \
    --memory 8Gi \
    --cpu 4 \
    --task-timeout 60m \
    --set-env-vars="NVIDIA_VISIBLE_DEVICES=all"

# Important Cloud Run GPU flags (currently in preview/restrictive regions)
# The user might need to change regions (e.g., us-central1) or use CPU if quota is missing.
# If this fails locally due to GPU flags, run: gcloud components update

echo "✅ Deployment Complete!"
echo "MoneyPrinter Training is now available natively inside the GCP Console!"
echo "You can trigger training anytime by executing:"
echo "➡️ gcloud run jobs execute $JOB_NAME --region $REGION"
