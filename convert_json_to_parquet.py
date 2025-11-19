"""Convert downloaded openFDA JSON batches into flattened Parquet files."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

BATCH_PATTERN = re.compile(r"batch_(\d+)\.json$")


def load_json_auto(path: Path) -> pd.DataFrame:
    """Load newline-delimited JSON if possible, fall back to standard JSON."""
    try:
        df = pd.read_json(path, lines=True)
    except ValueError:
        df = pd.read_json(path)
    records = df.to_dict(orient="records")
    return pd.json_normalize(records)


def convert_batches(
    input_dir: Path | str = "data/meta",
    output_dir: Path | str = "data/clean/parquet",
    glob_pattern: str = "batch_*.json",
) -> int:
    """Convert every JSON file that matches *glob_pattern* into Parquet."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {in_path}")

    out_path.mkdir(parents=True, exist_ok=True)

    matches: list[tuple[int | None, Path]] = []
    for file_path in sorted(in_path.glob(glob_pattern)):
        match = BATCH_PATTERN.match(file_path.name)
        batch_idx = int(match.group(1)) if match else None
        matches.append((batch_idx, file_path))

    if not matches:
        print("No JSON files found to convert.")
        return 0

    matches.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else x[1].name))

    converted = 0
    for batch_idx, json_file in matches:
        print(f"Converting {json_file}...")
        df = load_json_auto(json_file)
        if df.empty:
            print(f"  Skipping {json_file.name}: no records present.")
            continue

        if batch_idx is not None:
            out_file = out_path / f"batch_{batch_idx}.parquet"
        else:
            out_file = out_path / f"{json_file.stem}.parquet"

        df.to_parquet(out_file, index=False)
        print(f"  Saved {out_file}")
        converted += 1

    print(f"Finished converting {converted} file(s).")
    return converted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/meta", help="Directory holding JSON batches")
    parser.add_argument("--output-dir", default="data/clean/parquet", help="Where to store Parquet files")
    parser.add_argument("--glob", default="batch_*.json", help="Glob used to match batch files")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    convert_batches(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        glob_pattern=args.glob,
    )
