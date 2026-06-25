"""
Upload a local folder to Azure Blob Storage, preserving the directory structure.

Usage:
    python uploadtoblob.py [options]

Requirements:
    pip install azure-storage-blob python-dotenv

Environment variables (or .env file):
    AZURE_STORAGE_CONNECTION_STRING  — connection string for the storage account
    AZURE_STORAGE_CONTAINER_NAME     — target container name
"""

import os
import sys
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()


def upload_folder_to_blob(
    local_folder: str,
    connection_string: str,
    container_name: str,
    blob_prefix: str = "",
    overwrite: bool = True,
) -> None:
    folder_path = Path(local_folder).resolve()
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    client = BlobServiceClient.from_connection_string(connection_string)
    container_client = client.get_container_client(container_name)

    # Create container if it doesn't exist
    if not container_client.exists():
        container_client.create_container()
        print(f"Created container: {container_name}")

    files = [f for f in folder_path.rglob("*") if f.is_file()]
    if not files:
        print("No files found in the specified folder.")
        return

    print(f"Uploading {len(files)} file(s) from '{folder_path}' to container '{container_name}'...")

    uploaded = 0
    failed = 0

    for file_path in files:
        # Preserve the folder structure relative to the source folder
        relative_path = file_path.relative_to(folder_path)
        blob_name = "/".join(relative_path.parts)  # always use forward slashes for blob paths
        if blob_prefix:
            blob_name = f"{blob_prefix.rstrip('/')}/{blob_name}"

        try:
            blob_client = container_client.get_blob_client(blob_name)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=overwrite)
            print(f"  [OK] {relative_path}  →  {blob_name}")
            uploaded += 1
        except Exception as exc:
            print(f"  [FAIL] {relative_path}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone. {uploaded} uploaded, {failed} failed.")
    if failed:
        sys.exit(1)


LOCAL_FOLDER = r"C:\D\SampleData"
CONTAINER_NAME = "source"


def main() -> None:
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if not connection_string:
        print(
            "Error: Azure Storage connection string is required. "
            "Set AZURE_STORAGE_CONNECTION_STRING in your environment or .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    upload_folder_to_blob(
        local_folder=LOCAL_FOLDER,
        connection_string=connection_string,
        container_name=CONTAINER_NAME,
    )


if __name__ == "__main__":
    main()
