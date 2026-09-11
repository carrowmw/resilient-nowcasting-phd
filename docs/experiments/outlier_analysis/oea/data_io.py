"""Data acquisition: cached, concurrent pulls from Cloudflare R2 + Weights & Biases.

R2 objects are public, so we fetch over plain HTTPS (no signing needed) and cache
each JSON payload under ``cache/{run_id}/...`` mirroring the R2 key layout.  Re-runs
are cheap; pass ``refresh=True`` to bypass the cache.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from . import config

_TIMEOUT = 60
_RETRIES = 8


# --------------------------------------------------------------------------- #
# Low-level HTTP with on-disk cache
# --------------------------------------------------------------------------- #
def _http_get_json(url: str) -> Any:
    last_err: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "oea/1.0"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code == 404:
                break  # definitive: object does not exist
            if exc.code == 429:  # rate-limited: honour Retry-After / back off hard
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(retry_after) if retry_after else min(2.0 * 2 ** attempt, 60)
                time.sleep(delay)
                continue
            time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {_RETRIES} tries: {url} ({last_err})")


def _cached_json(rel_key: str, cache_dir: Path, refresh: bool) -> Any | None:
    """Fetch ``{R2}/{rel_key}``, caching under ``{cache_dir}/{rel_key}``.

    Returns ``None`` if the object does not exist (HTTP 404) — some node/date
    combinations legitimately have no payload.
    """
    dest = cache_dir / rel_key
    if dest.exists() and not refresh:
        try:
            return json.loads(dest.read_text())
        except json.JSONDecodeError:
            pass  # corrupt cache -> re-download

    url = f"{config.R2_PUBLIC_URL}/{rel_key}"
    try:
        data = _http_get_json(url)
    except RuntimeError as exc:
        if "404" in str(exc):
            return None
        raise
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data))
    return data


# --------------------------------------------------------------------------- #
# Registry / spatial
# --------------------------------------------------------------------------- #
def load_registry(refresh: bool = False) -> dict:
    return _cached_json("runs/run_registry.json", config.CACHE_DIR, refresh)


def load_spatial(run_id: str, refresh: bool = False) -> dict:
    return _cached_json(f"runs/{run_id}/spatial.json", config.CACHE_DIR, refresh)


def load_page(run_id: str, date: str, refresh: bool = False) -> dict | None:
    return _cached_json(f"runs/{run_id}/{date}/page.json", config.CACHE_DIR, refresh)


def load_node(run_id: str, date: str, node_id: int, refresh: bool = False) -> dict | None:
    return _cached_json(
        f"runs/{run_id}/{date}/nodes/node_{node_id}.json", config.CACHE_DIR, refresh
    )


# --------------------------------------------------------------------------- #
# Bulk concurrent prefetch
# --------------------------------------------------------------------------- #
def prefetch(
    run_id: str,
    dates: Iterable[str],
    node_ids: Iterable[int],
    *,
    max_workers: int = 24,
    refresh: bool = False,
    fetch_pages: bool = True,
    fetch_nodes: bool = True,
) -> dict[str, int]:
    """Warm the cache for every (date) page and (date, node) file, concurrently.

    Returns a small stats dict.  Progress is printed to stderr.
    """
    dates = list(dates)
    node_ids = list(node_ids)
    jobs: list[tuple[str, tuple]] = []
    if fetch_pages:
        jobs += [("page", (run_id, d)) for d in dates]
    if fetch_nodes:
        jobs += [("node", (run_id, d, n)) for d in dates for n in node_ids]

    total = len(jobs)
    stats = {"ok": 0, "missing": 0, "total": total}
    print(f"[data_io] prefetching {total} objects "
          f"({len(dates)} dates x {len(node_ids)} nodes) ...", file=sys.stderr)

    def _run(job):
        kind, args = job
        if kind == "page":
            return kind, load_page(*args, refresh=refresh)
        return kind, load_node(*args, refresh=refresh)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for fut in as_completed(futs):
            _, payload = fut.result()
            stats["ok" if payload is not None else "missing"] += 1
            done += 1
            if done % 200 == 0 or done == total:
                print(f"[data_io]   {done}/{total} "
                      f"(missing={stats['missing']})", file=sys.stderr)
    return stats


# --------------------------------------------------------------------------- #
# Weights & Biases training curves + config
# --------------------------------------------------------------------------- #
def load_wandb_run(wandb_run_id: str) -> dict | None:
    """Return training history + summary + config for one W&B run.

    Cached as JSON so the analysis is reproducible offline.  Returns ``None`` if
    W&B is unreachable or the run is missing (the report degrades gracefully).
    """
    cache_path = config.CACHE_DIR / "wandb" / f"{wandb_run_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    try:
        import wandb  # noqa: WPS433 (optional heavy dep)
    except ImportError:
        print("[data_io] wandb not installed; skipping W&B pull", file=sys.stderr)
        return None

    try:
        api = wandb.Api(timeout=90)
        run = api.run(f"{config.WANDB_ENTITY}/{config.WANDB_PROJECT}/{wandb_run_id}")
        keys = [
            "epoch", "train_loss_total", "train_loss_impute", "train_loss_forecast",
            "train_loss_kriging", "val_loss_total", "val_loss_kriging", "val_rmse",
            "val_mae", "val_zero_acc", "val_impute_mse", "val_kriging_rmse",
            "val_kriging_mae", "val_kriging_msle", "val_kriging_nodes",
        ]
        hist = run.history(keys=keys, pandas=False)
        summary = {k: v for k, v in dict(run.summary).items()
                   if not k.startswith("_") and isinstance(v, (int, float, str, bool))}
        out = {
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "config": dict(run.config),
            "summary": summary,
            "history": list(hist),
        }
    except Exception as exc:  # noqa: BLE001 - network/auth failures are non-fatal
        print(f"[data_io] W&B pull failed for {wandb_run_id}: {exc}", file=sys.stderr)
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out))
    return out
