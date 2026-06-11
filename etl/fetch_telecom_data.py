import requests
import pandas as pd
import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

# ── Load .env file ─────────────────────────────────────────────
load_dotenv()

# ── Config ─────────────────────────────────────────────────────
API_KEY     = os.getenv("DATA_GOV_API_KEY")
RESOURCE_ID = os.getenv("DATA_GOV_RESOURCE_ID")
BASE_URL    = "https://api.data.gov.in/resource"
CSV_FILE    = "telecom_data.csv"

# ── Safety check ───────────────────────────────────────────────
if not API_KEY or not RESOURCE_ID:
    raise ValueError("Missing API key or Resource ID in .env file!")

print(f"API Key loaded: {API_KEY[:10]}...")
print(f"Resource ID  : {RESOURCE_ID}")

# ── MongoDB ─────────────────────────────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db     = client["telecom_churn_db"]

# ══════════════════════════════════════════════════════════════
# STEP 1: Fetch ALL pages from data.gov.in API
# ══════════════════════════════════════════════════════════════
def fetch_all_data():
    all_records = []
    offset      = 0
    limit       = 500
    total       = None

    print("\nFetching data from data.gov.in ...")

    while True:
        params = {
            "api-key":     API_KEY,
            "format":      "json",
            "limit":       limit,
            "offset":      offset,
            "resource_id": RESOURCE_ID,
        }

        response = requests.get(BASE_URL, params=params, timeout=30)

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()

        if total is None:
            total = int(data.get("total", 0))
            print(f"Total records available: {total}")

        records = data.get("records", [])
        if not records:
            break

        all_records.extend(records)
        offset += limit

        print(f"Fetched {len(all_records)} / {total} records ...")

        if len(all_records) >= total:
            break

    print(f"\nTotal fetched: {len(all_records)} records")
    return all_records

# ══════════════════════════════════════════════════════════════
# STEP 2: Clean & Save as CSV
# ══════════════════════════════════════════════════════════════
def save_to_csv(records):
    if not records:
        print("No records to save!")
        return None

    df = pd.DataFrame(records)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    df.dropna(axis=1, how='all', inplace=True)

    df.to_csv(CSV_FILE, index=False)
    print(f"\nCSV saved : {CSV_FILE}")
    print(f"Rows      : {len(df)}")
    print(f"Columns   : {list(df.columns)}")
    return df

# ══════════════════════════════════════════════════════════════
# STEP 3: Store CSV data into MongoDB
# ══════════════════════════════════════════════════════════════
def store_to_mongodb(df):
    if df is None or df.empty:
        print("No data to insert!")
        return

    collection = db["telecom_raw_data"]
    collection.drop()
    print("\nOld collection dropped. Inserting fresh data ...")

    records = df.to_dict(orient="records")

    for rec in records:
        rec["inserted_at"] = datetime.now()

    result = collection.insert_many(records)
    print(f"Inserted {len(result.inserted_ids)} records into MongoDB")
    print(f"Collection: telecom_raw_data")

# ══════════════════════════════════════════════════════════════
# STEP 4: Check operator-wise summary
# ══════════════════════════════════════════════════════════════
def show_operator_summary():
    collection = db["telecom_raw_data"]
    total = collection.count_documents({})
    print(f"\nTotal documents in MongoDB: {total}")

    sample = collection.find_one({}, {'_id': 0})
    if sample:
        print(f"\nSample record keys : {list(sample.keys())}")
        print(f"\nSample record      :\n{sample}")

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    records = fetch_all_data()
    df = save_to_csv(records)
    store_to_mongodb(df)
    show_operator_summary()

    print("\nDone! Your MongoDB is restored.")