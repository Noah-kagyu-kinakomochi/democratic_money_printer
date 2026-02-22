# 🛠️ MoneyPrinter — Setup Checklist

This is a step-by-step guide of all the **manual** actions you must take before the system is fully operational on GCP.

---

## Phase 1 — Google Cloud Platform Setup

### Step 1: Create a GCP Project
1. Go to [https://console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project (e.g., `moneyprinter-prod`)
3. Note your **Project ID** — you'll need this in several places
user input: moneyprinter-prod created in google cloud free trial! 90days 300dollars free cash
### Step 2: Enable Required GCP APIs
In **APIs & Services → Enable APIs**, enable all of these:
- Cloud Storage API
- Compute Engine API
- IAM API

Or run this in Cloud Shell:
```bash
gcloud services enable compute.googleapis.com storage.googleapis.com iam.googleapis.com
```

### Step 3: Create a GCS Bucket
This bucket stores your training dataset and trained model artifacts.
```bash
gsutil mb -l us-central1 gs://moneyprinter-artifacts-prod
# ✅ Already created: gs://moneyprinter-artifacts-prod
```

### Step 4: Create a Service Account for GitHub Actions
```bash
# Create the service account
gcloud iam service-accounts create moneyprinter-trainer \
  --description="MoneyPrinter GCP Training SA" \
  --display-name="MoneyPrinter Trainer"

# Grant GCS access (to read/write training data and models)
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:moneyprinter-trainer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Grant Compute access (to create/delete GPU VMs)
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:moneyprinter-trainer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/compute.instanceAdmin.v1"
```

### Step 5: Set Up Workload Identity Federation (Recommended)
This allows GitHub Actions to authenticate securely without a JSON key file.
> Follow the official guide: https://cloud.google.com/blog/products/identity-security/enabling-keyless-authentication-from-github-actions  
> **OR** as a quick alternative (less secure): download a JSON key and add it to GitHub Secrets.

---

## Phase 2 — GitHub Secrets

Go to your GitHub repo → **Settings → Secrets and Variables → Actions** and add these secrets:

| Secret Name | Value |
|---|---|
| `WORKLOAD_IDENTITY_PROVIDER` | From Step 5 (format: `projects/…/providers/…`) |
| `GCP_SERVICE_ACCOUNT` | `moneyprinter-trainer@YOUR_PROJECT_ID.iam.gserviceaccount.com` |

---

## Phase 3 — Update the Workflow File

Open [`.github/workflows/gcp-gpu-training.yml`](file:///Users/hssetasosan/moneyprinter/.github/workflows/gcp-gpu-training.yml) and update the `env` block at the top:

```yaml
env:
  PROJECT_ID: moneyprinter-prod           # ✅ Set
  ZONE: us-central1-a
  BUCKET_NAME: moneyprinter-artifacts-prod # ✅ Set
  TRAINING_SCRIPT: training_bundle/train.py
```

---

## Phase 4 — Local Environment

Your local machine and/or the server running the daily execution loop need these environment variables set (add to `.env`):

```env
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
TRADING_MODE=paper
GCS_BUCKET_NAME=moneyprinter-artifacts-prod  # ✅ Bucket is ready
# GOOGLE_APPLICATION_CREDENTIALS not needed — using gcloud ADC
```

---

## Phase 5 — First Training Run

Once all of the above is done, kick off the first training run:

```bash
# 1. Ingest market data locally first
python main.py ingest

# 2. Package data and upload directly to GCS
python main.py package

# 3. Push to main branch to trigger GitHub Actions (or trigger manually)
git push origin main
# --- OR go to GitHub Actions tab and click "Run workflow" ---
```

After the workflow completes, `best_model.pth` and `scaler.pkl` will be at `gs://your-bucket/models/` and the bot will automatically download them on its next startup.

---

## Phase 6 — Daily Execution Server

> ⚠️ **Do NOT run the daily execution loop on your laptop or Raspberry Pi.** A home internet dropout while holding a leveraged position can result in a loss from which you cannot recover.

**Recommended options (in order of preference):**
1. **GCP `e2-micro`** — Free tier, always-on, runs `python main.py run` on a cron via `cron` or Cloud Scheduler.
2. **AWS EC2 `t2.micro`** — Also free-tier eligible.
3. **Cloud Run** — Serverless, but requires containerizing the bot with Docker.

**Minimal cron setup on an always-on VM:**
```bash
# Edit cron (runs the bot every 30 minutes on weekdays)
crontab -e

# Add this line:
*/30 9-16 * * 1-5 cd /home/your_user/moneyprinter && python main.py run >> /var/log/moneyprinter.log 2>&1
```

---

## Summary Checklist

- [x] **GCP**: Create project, enable APIs
- [x] **GCS**: Create storage bucket (`moneyprinter-artifacts-prod`)
- [x] **IAM**: Create service account with correct roles (`moneyprinter-trainer`)
- [ ] **Auth**: Set up Workload Identity Federation (GitHub Actions → GCP)
- [ ] **GitHub Secrets**: Add `WORKLOAD_IDENTITY_PROVIDER` and `GCP_SERVICE_ACCOUNT`
- [x] **Workflow YAML**: Updated `PROJECT_ID` and `BUCKET_NAME`
- [ ] **Local `.env`**: Add `GCS_BUCKET_NAME=moneyprinter-artifacts-prod` and Alpaca keys
- [ ] **First Run**: `python main.py ingest && python main.py package`, then trigger GitHub Action
- [ ] **Daily Server**: Deploy to a cloud VM with a cron schedule
