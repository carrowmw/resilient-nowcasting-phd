#!/usr/bin/env python3
"""
Migrate registry.json to the current config SHA.

When model blocks are added to a quality config (e.g. quality_full_v2.yaml),
the config SHA changes. The experiment runner uses SHA-keyed skip logic:

    if registry.get(job_key) == sha: skip

Without migration, every existing completed job would re-run because the
stored SHA no longer matches the new one.  This script back-fills all
existing entries with the new SHA so only genuinely new job keys run.

Usage (inside the container, after editing the config):
    python scripts/migrate_registry_sha.py \\
        --config configs/quality/quality_full_v2.yaml \\
        --registry artifacts/quality_full_v2/registry.json

    # Dry-run (no writes):
    python scripts/migrate_registry_sha.py \\
        --config configs/quality/quality_full_v2.yaml \\
        --registry artifacts/quality_full_v2/registry.json \\
        --dry-run
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


def config_sha(config: dict) -> str:
    canonical = yaml.safe_dump(config, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the updated YAML config")
    parser.add_argument("--registry", required=True, help="Path to registry.json")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    registry_path = Path(args.registry)

    assert config_path.exists(), f"CRITICAL: config not found: {config_path}"
    assert registry_path.exists(), f"CRITICAL: registry not found: {registry_path}"

    with open(config_path) as f:
        config = yaml.safe_load(f)
    new_sha = config_sha(config)

    registry = json.loads(registry_path.read_text())

    old_shas = set(registry.values())
    assert len(old_shas) == 1, (
        f"CRITICAL: registry contains entries from multiple SHAs {old_shas} — "
        "manual inspection required before migration"
    )
    old_sha = next(iter(old_shas))

    if old_sha == new_sha:
        print(f"SHA unchanged ({new_sha}). Nothing to migrate.")
        sys.exit(0)

    print(f"Old SHA: {old_sha}")
    print(f"New SHA: {new_sha}")
    print(f"Entries to update: {len(registry)}")

    if args.dry_run:
        print("DRY RUN — no changes written.")
        sys.exit(0)

    updated = {k: new_sha for k in registry}
    registry_path.write_text(json.dumps(updated, indent=2))
    print(f"Done. Registry updated at {registry_path}")


if __name__ == "__main__":
    main()
