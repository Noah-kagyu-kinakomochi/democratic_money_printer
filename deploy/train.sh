#!/bin/bash
set -e

# ==============================================================================
# MoneyPrinter: Trigger 'Train' Job (Deep Learning on GCP GPU)
# 
# This script securely provisions an ephemeral NVIDIA Tesla T4 GPU on
# Google Compute Engine. The VM downloads your packaged training data from GCS,
# trains the Deep Learning model, uploads the new weights to GCS, and
# immediately DESTROYS ITSELF to save your billing credits.
# ==============================================================================

PROJECT_ID=$(gcloud config get-value project)
ZONE="us-central1-a"
BUCKET_NAME="moneyprinter-artifacts-${PROJECT_ID#moneyprinter-}"
REPO_NAME="moneyprinter-repo"

# Generate a unique instance name to avoid collisions
INSTANCE_NAME="gpu-trainer-$(date +%s)"
SERVICE_ACCOUNT="moneyprinter-trainer@${PROJECT_ID}.iam.gserviceaccount.com"

echo "🚀 Triggering MoneyPrinter 'Train' Pipeline on Ephemeral GPU..."
echo "Project: $PROJECT_ID"
echo "VM Name: $INSTANCE_NAME"
echo "Bucket : $BUCKET_NAME"

# We pass this script directly into the VM to execute on boot
STARTUP_SCRIPT=$(cat <<EOF
#!/bin/bash
set -e
exec > /var/log/moneyprinter-train.log 2>&1

echo "==========================================="
echo "🖨️💰 MoneyPrinter GPU Training Initiated"
echo "==========================================="

cd /home/appuser

# 1. Download packaged data from GCS
echo "Downloading dataset from gs://${BUCKET_NAME}..."
mkdir -p training_bundle results
gsutil cp gs://${BUCKET_NAME}/datasets/data.parquet training_bundle/data.parquet

# 2. Clone the repository so we have the training scripts
echo "Fetching training scripts from git..."
git clone https://github.com/Noah-kagyu-kinakomochi/democratic_money_printer.git repo
cd repo

# 3. Setup Python Environment (bypassing PEP-668)
echo "Installing dependencies..."
pip3 install --break-system-packages -r training_bundle/requirements_train.txt

# 4. Execute the training run
echo "Running PyTorch training..."
python3 training_bundle/train.py --data ../training_bundle/data.parquet --output ../results

# 5. Upload New Artifacts back to GCS
echo "Uploading model & scaler to GCS..."
gsutil cp ../results/best_model.pth gs://${BUCKET_NAME}/models/best_model.pth
gsutil cp ../results/scaler.pkl gs://${BUCKET_NAME}/models/scaler.pkl

echo "==========================================="
echo "✅ Training Complete. Self-Destructing VM..."
echo "==========================================="

# Automatically delete the instance saving billing credits
gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE} --quiet
EOF
)

# Create the VM instance and pass the startup script
echo "☁️  Provisioning NVIDIA T4 Spot Instance..."
gcloud compute instances create ${INSTANCE_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata="install-nvidia-driver=True,startup-script=${STARTUP_SCRIPT}" \
    --service-account=${SERVICE_ACCOUNT}

echo "✅ Deployment Triggered!"
echo "The GPU instance '${INSTANCE_NAME}' is now booting."
echo "You can monitor the serial console logs in GCP Console to watch the training progress."
echo "The VM will automatically DELETE ITSELF as soon as training finishes."
