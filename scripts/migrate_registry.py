#!/usr/bin/env python3
"""
One-shot registry migration: remaps existing job keys from the old config SHA
to the new config SHA after adding Transformer models to the config.

This ensures that re-running the experiment runner skips all previously
completed jobs and only trains the new Transformer models.

Usage (inside the container):
    python scripts/migrate_registry.py --config configs/quality/quality_full_v2.yaml
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
    parser.add_argument("--config", required=True, help="Path to the updated config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    new_sha = config_sha(config)
    out_dir = Path(config["run"]["output_dir"])
    registry_path = out_dir / "registry.json"

    if not registry_path.exists():
        print(f"No registry found at {registry_path} — nothing to migrate.")
        sys.exit(0)

    registry = json.loads(registry_path.read_text())
    old_shas = set(registry.values())

    if new_sha in old_shas:
        print(f"Registry already uses the new SHA ({new_sha}). No migration needed.")
        sys.exit(0)

    migrated = 0
    for key in registry:
        old_val = registry[key]
        if old_val != new_sha:
            registry[key] = new_sha
            migrated += 1

    print(f"New config SHA: {new_sha}")
    print(f"Old SHA(s) found: {old_shas}")
    print(f"Migrated {migrated} entries.")

    if args.dry_run:
        print("[DRY RUN] No changes written.")
    else:
        registry_path.write_text(json.dumps(registry, indent=2))
        print(f"Registry updated at {registry_path}")


if __name__ == "__main__":
    main()
