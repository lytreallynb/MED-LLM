import os
import json
import time
import requests
from tqdm import tqdm

API_URL = "https://api.fda.gov/drug/label.json"
SAVE_META_DIR = "data/meta"
SAVE_ZIP_DIR = "data/raw"
BATCH_SIZE = 100  # FDA allows up to 100 per request

os.makedirs(SAVE_META_DIR, exist_ok=True)
os.makedirs(SAVE_ZIP_DIR, exist_ok=True)

def fetch_batch(skip):
    """Fetch a batch of drug labels from openFDA"""
    params = {"limit": BATCH_SIZE, "skip": skip}

    r = requests.get(API_URL, params=params, timeout=20)
    if r.status_code != 200:
        print("Error:", r.text)
        return None

    return r.json()


def extract_download_url(record):
    """FDA stores zipped drug SPL files at a stable path"""
    if "id" not in record:
        return None

    doc_id = record["id"]
    return f"https://download.open.fda.gov/flatfiles/drug/label/{doc_id}.zip"


def download_zip(url):
    filename = os.path.join(SAVE_ZIP_DIR, url.split("/")[-1])

    if os.path.exists(filename):
        return  # skip existing

    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with open(filename, "wb") as f:
                f.write(r.content)
        else:
            print("Failed:", url, r.status_code)
    except Exception as e:
        print("Error downloading:", url, e)


def main():
    skip = 0
    meta_index = 0

    while True:
        print(f"Fetching batch: skip={skip}")

        data = fetch_batch(skip)
        if not data or "results" not in data:
            print("No more data. Done.")
            break

        results = data["results"]
        if len(results) == 0:
            print("All done.")
            break

        # Save metadata of this batch
        meta_path = os.path.join(SAVE_META_DIR, f"batch_{meta_index}.json")
        with open(meta_path, "w") as f:
            json.dump(results, f, indent=2)

        # Download ZIP files
        for record in tqdm(results, desc="Downloading ZIPs"):
            url = extract_download_url(record)
            if url:
                download_zip(url)

        skip += BATCH_SIZE
        meta_index += 1
        time.sleep(0.2)  # avoid API rate limits


if __name__ == "__main__":
    main()
