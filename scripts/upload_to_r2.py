"""
upload_to_r2.py

Upload a generated paginated-payload tree to Cloudflare R2 (S3-compatible object
storage) so the Next.js frontend can fetch payloads from R2_PUBLIC_URL instead of
local static files (Phase 3 of the database strategy).

Each object is gzip-compressed and uploaded with:
    ContentType:     application/json
    ContentEncoding: gzip
    CacheControl:    public, max-age=...
so browser `fetch()` transparently decompresses the response — no frontend decode
code required. The on-disk JSON keys map 1:1 to object keys, e.g.

    frontend/public/data/runs/dark-brook-37/spatial.json
        → s3://{bucket}/runs/dark-brook-37/spatial.json
        → {R2_PUBLIC_URL}/runs/dark-brook-37/spatial.json

Credentials are read from the environment (.env is auto-loaded):
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_PUBLIC_URL

Usage:
    python scripts/upload_to_r2.py [--runs-dir frontend/public/data/runs] [--prefix runs]
                                   [--dry-run] [--no-gzip] [--delete-extra]
"""

import argparse
import gzip
import logging
import mimetypes
import os
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("upload_to_r2")

DEFAULT_RUNS_DIR = "frontend/public/data/runs"
DEFAULT_PREFIX = "runs"
CACHE_CONTROL = "public, max-age=3600"


def load_env() -> None:
    """Load .env from the repo root (mirrors export_ui_payload's loader)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


def require_env(*keys: str) -> dict:
    vals = {k: os.getenv(k) for k in keys}
    missing = [k for k, v in vals.items() if not v]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")
    return vals


def make_client(account_id: str, access_key: str, secret_key: str):
    import boto3
    from botocore.config import Config
    endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    logger.info(f"R2 endpoint: {endpoint}")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )


def content_type_for(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def upload_tree(client, bucket: str, runs_dir: str, prefix: str,
                *, gzip_enabled: bool, dry_run: bool,
                public_url: Optional[str]) -> int:
    root = Path(runs_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"runs-dir not found: {runs_dir}")

    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"No files under {runs_dir}")

    n_bytes_in = n_bytes_out = 0
    for i, path in enumerate(files, 1):
        rel = path.relative_to(root).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        raw = path.read_bytes()
        n_bytes_in += len(raw)

        cache_ctrl = "public, max-age=120" if path.name == "run_registry.json" else CACHE_CONTROL
        extra = {"ContentType": content_type_for(path), "CacheControl": cache_ctrl}
        body = raw
        if gzip_enabled:
            body = gzip.compress(raw)
            extra["ContentEncoding"] = "gzip"
        n_bytes_out += len(body)

        if dry_run:
            if i <= 5 or i == len(files):
                logger.info(f"[dry-run] would PUT {key}  ({len(raw)}→{len(body)} B)")
            continue
        client.put_object(Bucket=bucket, Key=key, Body=body, **extra)
        if i % 100 == 0 or i == len(files):
            logger.info(f"  uploaded {i}/{len(files)} …")

    logger.info(
        f"{'[dry-run] ' if dry_run else ''}Done: {len(files)} objects | "
        f"{n_bytes_in/1e6:.1f} MB → {n_bytes_out/1e6:.1f} MB "
        f"({'gzip' if gzip_enabled else 'raw'})"
    )
    if public_url:
        sample = f"{prefix}/run_registry.json" if prefix else "run_registry.json"
        logger.info(f"Public URL base: {public_url.rstrip('/')}/{sample}")
    return len(files)


def main() -> None:
    p = argparse.ArgumentParser(description="Upload paginated payloads to Cloudflare R2.")
    p.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    p.add_argument("--prefix", default=DEFAULT_PREFIX,
                   help="Key prefix in the bucket (default 'runs').")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-gzip", action="store_true", help="Upload raw (no gzip/ContentEncoding).")
    args = p.parse_args()

    load_env()
    env = require_env("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME")
    public_url = os.getenv("R2_PUBLIC_URL")

    client = None if args.dry_run else make_client(
        env["R2_ACCOUNT_ID"], env["R2_ACCESS_KEY_ID"], env["R2_SECRET_ACCESS_KEY"]
    )
    upload_tree(
        client, env["R2_BUCKET_NAME"], args.runs_dir, args.prefix,
        gzip_enabled=not args.no_gzip, dry_run=args.dry_run, public_url=public_url,
    )


if __name__ == "__main__":
    main()
