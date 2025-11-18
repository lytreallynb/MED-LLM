import os
import re
import pandas as pd

META_DIR = "data/meta"
PARQUET_DIR = "data/clean/parquet"

os.makedirs(PARQUET_DIR, exist_ok=True)

# Regex to extract number from filename
pattern = re.compile(r"batch_(\d+)\.json")

def load_json_auto(path):
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        return pd.read_json(path)

def convert_all_batches():
    # Load all matching files with numeric sort
    files = []
    for f in os.listdir(META_DIR):
        match = pattern.match(f)
        if match:
            idx = int(match.group(1))
            files.append((idx, f))

    if not files:
        print("No JSON batch files found.")
        return

    # Sort numerically by batch id
    files.sort(key=lambda x: x[0])

    print(f"Found {len(files)} JSON files")

    for idx, filename in files:
        in_path = os.path.join(META_DIR, filename)
        out_path = os.path.join(PARQUET_DIR, f"batch_{idx}.parquet")

        print(f"[{idx}] Converting {filename} → batch_{idx}.parquet")

        df = load_json_auto(in_path)
        df = pd.json_normalize(df.to_dict(orient="records"))

        df.to_parquet(out_path, index=False)

        print(f"    Saved {out_path}")

    print("All JSON files successfully converted.")

if __name__ == "__main__":
    convert_all_batches()
