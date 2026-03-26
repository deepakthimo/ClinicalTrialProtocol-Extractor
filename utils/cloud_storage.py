import os
import json
import logging
from typing import Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
load_dotenv()
import logging

logger = logging.getLogger(__name__)

class CloudStorageManager:
    def __init__(self):
        self.provider = "local"
        self.client = None
        self.bucket = None
        
        self.aws_bucket = os.getenv("AWS_S3_BUCKET_NAME")
        self.gcp_bucket = os.getenv("GCP_BUCKET_NAME")
        
        if self.aws_bucket:
            try:
                import boto3
                self.client = boto3.client('s3')
                self.provider = "aws"
                logger.info("☁️ Storage Manager initialized with AWS S3.")
            except ImportError:
                logger.error("boto3 is not installed. Cannot use AWS S3.")
            except Exception as e:
                logger.error(f"Failed to initialize AWS S3 client: {e}")
        
        elif self.gcp_bucket:
            try:
                from google.cloud import storage

                gcp_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if not gcp_creds_path:
                    logger.warning("⚠️ GOOGLE_APPLICATION_CREDENTIALS is not set in .env. "
                                   "GCP might fail unless running inside a Google Cloud environment.")
                elif not os.path.exists(gcp_creds_path):
                    logger.error(f"❌ GCP Credentials file not found at: {gcp_creds_path}")

                self.client = storage.Client()
                self.bucket = self.client.bucket(self.gcp_bucket)
                self.provider = "gcp"
                logger.info("☁️ Storage Manager initialized with Google Cloud Storage.")
            except ImportError:
                logger.error("google-cloud-storage is not installed. Cannot use GCP.")
            except Exception as e:
                logger.error(f"Failed to initialize GCP client: {e}")
                self.provider = "local"
        
        if self.provider == "local":
            logger.warning("No active cloud storage configuration found (missing AWS_S3_BUCKET_NAME or GCP_BUCKET_NAME). Falling back to local storage.")

    def is_enabled(self) -> bool:
        return self.provider != "local"

    def upload_json(self, object_name: str, data: Dict[str, Any]) -> str:
        """Uploads a dictionary as a JSON file to Cloud Storage and returns the URI."""
        json_str = json.dumps(data, indent=4)
        
        if self.provider == "aws":
            from botocore.exceptions import ClientError
            try:
                self.client.put_object(
                    Bucket=self.aws_bucket,
                    Key=object_name,
                    Body=json_str,
                    ContentType='application/json'
                )
                uri = f"s3://{self.aws_bucket}/{object_name}"
                logger.info(f"✅ Successfully uploaded JSON to {uri}")
                return uri
            except ClientError as e:
                logger.error(f"❌ Failed to upload {object_name} to S3: {e}")
                raise e
                
        elif self.provider == "gcp":
            try:
                blob = self.bucket.blob(object_name)
                blob.upload_from_string(json_str, content_type='application/json')
                uri = f"gs://{self.gcp_bucket}/{object_name}"
                logger.info(f"✅ Successfully uploaded JSON to {uri}")
                return uri
            except Exception as e:
                logger.error(f"❌ Failed to upload {object_name} to GCP: {e}")
                raise e
                
        else:
            logger.warning(f"Cloud storage disabled. Skipping cloud upload for {object_name}")
            return f"local://{object_name}"

    def download_json(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Downloads a JSON file from Cloud Storage and parses it."""
        if self.provider == "aws":
            from botocore.exceptions import ClientError
            try:
                response = self.client.get_object(Bucket=self.aws_bucket, Key=object_name)
                content = response['Body'].read().decode('utf-8')
                return json.loads(content)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code')
                if error_code == 'NoSuchKey':
                    return None
                logger.error(f"❌ Failed to download {object_name} from S3: {e}")
                return None
                
        elif self.provider == "gcp":
            from google.cloud.exceptions import NotFound
            try:
                blob = self.bucket.blob(object_name)
                content = blob.download_as_string()
                return json.loads(content)
            except NotFound:
                return None
            except Exception as e:
                logger.error(f"❌ Failed to download {object_name} from GCP: {e}")
                return None
                
        return None

storage_manager = CloudStorageManager()