from google.cloud import storage

BUCKET_NAME = "process-uncertainty-model"
BLOB_PATH = "data/math_completions.jsonl"
DEST_PATH = "data/math/qwen2.5-completions_v1.jsonl"

client = storage.Client()  # uses existing OAuth / gcloud auth
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob(BLOB_PATH)

blob.download_to_filename(DEST_PATH)

print(f"Downloaded to {DEST_PATH}")
