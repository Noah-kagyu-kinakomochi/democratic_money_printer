#!/bin/bash
set -e

# ==============================================================================
# MoneyPrinter: Deploy the 'Run' Job (Live Trading Execution)
# 
# This script builds the Docker container, pushes it to Google Artifact Registry,
# creates a Cloud Run Job, and sets up a Cloud Scheduler Cron to trigger it.
# ==============================================================================

# Variables
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
REPO_NAME="moneyprinter-repo"
IMAGE_NAME="moneyprinter-run"
JOB_NAME="moneyprinter-run-job"
SERVICE_ACCOUNT="moneyprinter-trainer@${PROJECT_ID}.iam.gserviceaccount.com"
CRON_SCHEDULE="0,15,30,45 * * * *" # Every 15 minutes (adjust to your strategy)
TIMEZONE="America/New_York"

echo "🚀 Deploying MoneyPrinter 'Run' Job to Google Cloud Run..."
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
    --args "main.py,run" \
    --service-account $SERVICE_ACCOUNT \
    --memory 512Mi \
    --cpu 1 \
    --task-timeout 5m

# 4. Set up Cloud Scheduler to trigger the Job
SCHEDULER_NAME="${JOB_NAME}-trigger"
if gcloud scheduler jobs describe $SCHEDULER_NAME --location=$REGION >/dev/null 2>&1; then
    echo "⏰ Updating Cloud Scheduler cron trigger: $SCHEDULER_NAME"
    gcloud scheduler jobs update http $SCHEDULER_NAME \
        --location=$REGION \
        --schedule="$CRON_SCHEDULE" \
        --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
        --http-method=POST \
        --oauth-service-account-email=$SERVICE_ACCOUNT
else
    echo "⏰ Creating Cloud Scheduler cron trigger: $SCHEDULER_NAME"
    gcloud scheduler jobs create http $SCHEDULER_NAME \
        --location=$REGION \
        --schedule="$CRON_SCHEDULE" \
        --time-zone="$TIMEZONE" \
        --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${JOB_NAME}:run" \
        --http-method=POST \
        --oauth-service-account-email=$SERVICE_ACCOUNT
fi

echo "✅ Deployment Complete! The MoneyPrinter 'Run' Job is now scheduled to execute every $CRON_SCHEDULE."
