"""
Run Azure Document Intelligence (prebuilt-layout) on all supported blobs in the
source container and write JSON results to the 'json' container.

Usage:
    python rundocintelligence.py

Requirements:
    pip install azure-ai-documentintelligence azure-storage-blob python-dotenv

Environment variables (or .env file):
    AZURE_STORAGE_CONNECTION_STRING   — connection string for the storage account
    AZURE_DOC_INTELLIGENCE_ENDPOINT   — Document Intelligence endpoint URL
    AZURE_DOC_INTELLIGENCE_KEY        — Document Intelligence API key
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas
from dotenv import load_dotenv

load_dotenv()

SOURCE_CONTAINER = "source"   # container where files were uploaded
JSON_CONTAINER = "json"                       # container to store JSON results
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
OVERWRITE = False   # set True to reprocess blobs that already have a JSON result


def _sas_url(account_name: str, account_key: str, container: str, blob_name: str) -> str:
    token = generate_blob_sas(
        account_name=account_name,
        container_name=container,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    return f"https://{account_name}.blob.core.windows.net/{container}/{blob_name}?{token}"


def process_blobs() -> None:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    endpoint = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")

    if not connection_string:
        print("Error: AZURE_STORAGE_CONNECTION_STRING not set.", file=sys.stderr)
        sys.exit(1)
    if not endpoint or not key:
        print("Error: AZURE_DOC_INTELLIGENCE_ENDPOINT and AZURE_DOC_INTELLIGENCE_KEY must be set.", file=sys.stderr)
        sys.exit(1)

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    account_name = blob_service.account_name
    account_key = blob_service.credential.account_key

    source_client = blob_service.get_container_client(SOURCE_CONTAINER)
    json_client = blob_service.get_container_client(JSON_CONTAINER)

    if not json_client.exists():
        json_client.create_container()
        print(f"Created container: {JSON_CONTAINER}")

    doc_client = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    blobs = [
        b for b in source_client.list_blobs()
        if any(b.name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)
    ]

    if not blobs:
        print("No supported files found in source container.")
        return

    print(f"Found {len(blobs)} file(s) to process...")

    processed = skipped = failed = 0

    for blob in blobs:
        json_blob_name = blob.name + ".json"

        if not OVERWRITE and json_client.get_blob_client(json_blob_name).exists():
            print(f"  [SKIP] {blob.name} (already has JSON)")
            skipped += 1
            continue

        try:
            url = _sas_url(account_name, account_key, SOURCE_CONTAINER, blob.name)

            poller = doc_client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(url_source=url),
            )
            result = poller.result()

            json_bytes = json.dumps(result.as_dict(), indent=2, ensure_ascii=False).encode("utf-8")
            json_client.get_blob_client(json_blob_name).upload_blob(
                json_bytes, overwrite=True, content_type="application/json"
            )
            print(f"  [OK] {blob.name}  →  {JSON_CONTAINER}/{json_blob_name}")
            processed += 1

        except Exception as exc:
            print(f"  [FAIL] {blob.name}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone. {processed} processed, {skipped} skipped, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    process_blobs()
