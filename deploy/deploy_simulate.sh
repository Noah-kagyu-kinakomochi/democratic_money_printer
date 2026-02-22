#!/bin/bash
set -e

# ==============================================================================
# MoneyPrinter: Deploy the 'Simulate' Job (Backtesting / Analysis)
# 
# This script builds the Docker container, pushes it to Google Artifact Registry,
# and creates a Cloud Run Job. It differs from the Run job because it runs 
# MANUALLY (no cron scheduler) and is given a larger memory space (2GB) for 
# iterating over pandas dataframes across thousands of rows.
# ==============================================================================

# Variables
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
REPO_NAME="moneyprinter-repo"
IMAGE_NAME="moneyprinter-simulate"
JOB_NAME="moneyprinter-simulate-job"
SERVICE_ACCOUNT="moneyprinter-trainer@${PROJECT_ID}.iam.gserviceaccount.com"

echo "🚀 Deploying MoneyPrinter 'Simulate' Job to Google Cloud Run..."
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

# 2. Build and Push the Docker Image using Cloud Build (so it builds native to GCP)
IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest"
echo "🔨 Building Docker image using Cloud Build..."
gcloud builds submit --tag $IMAGE_URL .

# 3. Create or Update the Cloud Run Job
echo "☁️  Deploying Cloud Run Job: $JOB_NAME..."
gcloud run jobs deploy $JOB_NAME \
    --image $IMAGE_URL \
    --region $REGION \
    --tasks 1 \
    --max-retries 0 \
    --command "python" \
    --args "main.py,simulate" \
    --service-account $SERVICE_ACCOUNT \
    --memory 2048Mi \
    --cpu 2 \
    --task-timeout 60m

echo "✅ Deployment Complete!"
echo "You can trigger a backtest simulation anytime by executing:"
echo "➡️ gcloud run jobs execute $JOB_NAME --region $REGION"
