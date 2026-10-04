"""Create a bucket on the local MinIO if it does not exist yet (used by CI for the DVC remote)."""

import os
import sys

from minio import Minio


def main(bucket: str) -> None:
    client = Minio(
        os.environ["RAG_MINIO_ENDPOINT"],
        access_key=os.environ["RAG_MINIO_ACCESS_KEY"],
        secret_key=os.environ["RAG_MINIO_SECRET_KEY"],
        secure=False,
    )
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


if __name__ == "__main__":
    main(sys.argv[1])
