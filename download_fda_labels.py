"""Download adverse drug label records from the openFDA API in JSONL batches."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

DEFAULT_ENDPOINT = "https://api.fda.gov/drug/label.json"


def download_batches(
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    save_dir: Path | str = "data/meta",
    limit: int = 100,
    max_records: int = 10_000,
    delay: float = 0.25,
) -> tuple[int, int]:
    """Fetch paginated openFDA responses and persist them as newline-delimited JSON."""
    if limit <= 0:
        raise ValueError("API limit must be greater than zero")
    if max_records <= 0:
        raise ValueError("max_records must be greater than zero")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    skip = 0
    batch_id = 0
    total_records = 0

    while total_records < max_records:
        current_limit = min(limit, max_records - total_records)
        params = {"limit": current_limit, "skip": skip}
        print(f"Requesting records {skip} → {skip + current_limit}...")

        try:
            response = session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            break

        payload = response.json()
        results = payload.get("results", [])
        if not results:
            print("Stopped: no more results returned by the API.")
            break

        outfile = save_path / f"batch_{batch_id}.json"
        with outfile.open("w", encoding="utf-8") as fh:
            for record in results:
                fh.write(json.dumps(record))
                fh.write("\n")

        print(f"Saved {outfile}")

        batch_id += 1
        downloaded = len(results)
        total_records += downloaded
        skip += downloaded

        if delay > 0:
            time.sleep(delay)

    print(f"Completed. Batches: {batch_id} | Records: {total_records}")
    return batch_id, total_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="openFDA endpoint URL")
    parser.add_argument("--save-dir", default="data/meta", help="Directory for the JSON batches")
    parser.add_argument("--limit", type=int, default=100, help="Number of records per request (max 100)")
    parser.add_argument("--max-records", type=int, default=10_000, help="Stop after downloading this many records")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay in seconds between requests")
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    download_batches(
        endpoint=cli_args.endpoint,
        save_dir=cli_args.save_dir,
        limit=cli_args.limit,
        max_records=cli_args.max_records,
        delay=cli_args.sleep,
    )
