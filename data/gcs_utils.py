"""
Utility module for Google Cloud Storage interactions.
Handles uploading training data and downloading model artifacts.
"""

import logging
import os
from google.cloud import storage

logger = logging.getLogger(__name__)

class GCSManager:
    def __init__(self, bucket_name: str = None):
        """
        Initialize GCS client.
        Uses GOOGLE_APPLICATION_CREDENTIALS environment variable or default credentials.
        """
        self.bucket_name = bucket_name or os.environ.get("GCS_BUCKET_NAME")
        if not self.bucket_name:
            logger.warning("GCS_BUCKET_NAME not set. GCS operations may fail if bucket is not provided explicitly.")
            
        try:
            self.client = storage.Client()
        except Exception as e:
            logger.error(f"Failed to initialize GCS Client: {e}")
            self.client = None

    def upload_file(self, local_path: str, gcs_path: str, bucket_name: str = None) -> bool:
        """Uploads a local file to GCS."""
        if not self.client:
            logger.error("GCS Client not initialized. Cannot upload.")
            return False
            
        target_bucket = bucket_name or self.bucket_name
        if not target_bucket:
            logger.error("No GCS bucket specified for upload.")
            return False

        if not os.path.exists(local_path):
            logger.error(f"Local file not found: {local_path}")
            return False

        try:
            bucket = self.client.bucket(target_bucket)
            blob = bucket.blob(gcs_path)
            
            logger.info(f"Uploading {local_path} to gs://{target_bucket}/{gcs_path}...")
            blob.upload_from_filename(local_path)
            logger.info("✅ Upload complete.")
            return True
        except Exception as e:
            logger.error(f"GCS Upload failed: {e}")
            return False

    def download_file(self, gcs_path: str, local_path: str, bucket_name: str = None, ignore_missing: bool = False) -> bool:
        """Downloads a file from GCS to local path."""
        if not self.client:
            logger.error("GCS Client not initialized. Cannot download.")
            return False
            
        target_bucket = bucket_name or self.bucket_name
        if not target_bucket:
            logger.error("No GCS bucket specified for download.")
            return False

        try:
            bucket = self.client.bucket(target_bucket)
            blob = bucket.blob(gcs_path)
            
            if not blob.exists():
                msg = f"File gs://{target_bucket}/{gcs_path} does not exist."
                if ignore_missing:
                    logger.debug(msg)
                else:
                    logger.error(msg)
                return False

            # Ensure local directory exists
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            
            logger.info(f"Downloading gs://{target_bucket}/{gcs_path} to {local_path}...")
            blob.download_to_filename(local_path)
            logger.info("✅ Download complete.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to download {gcs_path}: {e}")
            return False

def get_secret(secret_id: str, project_id: str = None) -> str:
    """
    Fetch a secret from Google Cloud Secret Manager.
    Returns None if missing or if credentials fail.
    """
    try:
        from google.cloud import secretmanager
    except ImportError:
        logger.warning(f"google-cloud-secret-manager not installed. Cannot fetch {secret_id}.")
        return None

    if not project_id:
        return None
        
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.debug(f"Could not fetch secret {secret_id} from GCP: {e}")
        return None
