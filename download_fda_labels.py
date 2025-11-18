import os
import json
import time
import requests

# Output directory
SAVE_DIR = "data/meta"
os.makedirs(SAVE_DIR, exist_ok=True)

# API config
ENDPOINT = "https://api.fda.gov/drug/label.json"
LIMIT = 100
MAX_RECORDS = 50000  # customize if needed

def download_batches():
    skip = 0
    batch_id = 0
    total = 0

    while total < MAX_RECORDS:
        print(f"Requesting {skip} → {skip + LIMIT}...")

        response = requests.get(ENDPOINT, params={"limit": LIMIT, "skip": skip})
        if response.status_code != 200:
            print("Stopped: no more data or API error.")
            break

        results = response.json().get("results", [])
        if not results:
            print("Stopped: no more results.")
            break

        # Save JSONL format
        outpath = os.path.join(SAVE_DIR, f"batch_{batch_id}.json")
        with open(outpath, "w") as f:
            for record in results:
                f.write(json.dumps(record) + "\n")

        print(f"Saved batch_{batch_id}.json")
        batch_id += 1
        total += len(results)
        skip += LIMIT

        time.sleep(0.3)  # openFDA rate limit

    print(f"Completed. Total records downloaded: {total}")

if __name__ == "__main__":
    download_batches()
