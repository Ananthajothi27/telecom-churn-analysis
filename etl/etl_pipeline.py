import requests
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import os
import sys

# ── Configuration ─────────────────────────────────────────────
API_BASE_URL = "http://127.0.0.1:8000"
MONGO_URI    = "mongodb://localhost:27017/"
DB_NAME      = "telecom_churn_db"

# ── MongoDB Connection ─────────────────────────────────────────
def get_db():
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]
    print("✅ Connected to MongoDB!")
    return db

# ════════════════════════════════════════════════════════════════
#  STEP 1 — EXTRACT
#  Fetch raw dirty data from our Live API
# ════════════════════════════════════════════════════════════════
def extract():
    print("\n📥 STEP 1: EXTRACTING data from API...")
    try:
        
        response = requests.get(f"{API_BASE_URL}/customers", timeout=30)
        if response.status_code == 200:
            data      = response.json()
            customers = data['customers']
            print(f"✅ Extracted {len(customers)} customers from API!")
            return customers
        else:
            print(f"❌ API returned status code: {response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)

# ════════════════════════════════════════════════════════════════
#  STEP 2 — LOAD RAW DATA
#  Store dirty data as-is into MongoDB (raw collection)
# ════════════════════════════════════════════════════════════════
def load_raw(db, customers):
    print("\n📦 STEP 2: LOADING raw dirty data into MongoDB...")
    raw_collection = db['raw_customers']
    raw_collection.drop()  # fresh load every time

    # Add metadata to each record
    for customer in customers:
        customer['_extracted_at'] = datetime.now().isoformat()
        customer['_is_cleaned']   = False

    raw_collection.insert_many(customers)
    print(f"✅ {len(customers)} raw records loaded into 'raw_customers' collection!")
    return customers

# ════════════════════════════════════════════════════════════════
#  STEP 3 — TRANSFORM
#  Clean and fix all dirty data issues
# ════════════════════════════════════════════════════════════════
def transform(customers):
    print("\n🔄 STEP 3: TRANSFORMING data...")
    df = pd.DataFrame(customers)

    print(f"   📊 Total records before cleaning : {len(df)}")
    print(f"   ❌ Null values before cleaning   :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # ── Fix 1: Standardize gender ──────────────────────────────
    df['gender'] = df['gender'].str.capitalize()
    print("   ✅ Fix 1: Gender standardized (male→Male, female→Female)")

    # ── Fix 2: Fill missing age with median ───────────────────
    median_age   = df['age'].median()
    df['age']    = df['age'].fillna(median_age).astype(int)
    print(f"  ✅ Fix 2: Missing age filled with median ({median_age})")

    # ── Fix 3: Fill missing data_usage_gb with mean ───────────
    mean_data           = df['data_usage_gb'].mean()
    df['data_usage_gb'] = df['data_usage_gb'].fillna(round(mean_data, 2))
    print(f"  ✅ Fix 3: Missing data_usage_gb filled with mean ({round(mean_data,2)})")

    # ── Fix 4: Fill missing sms_per_day with 0 ───────────────
    df['sms_per_day'] = df['sms_per_day'].fillna(0)
    print("   ✅ Fix 4: Missing sms_per_day filled with 0")

    # ── Fix 5: Fill missing avg_call_duration_min with mean ──
    mean_call                    = df['avg_call_duration_min'].mean()
    df['avg_call_duration_min']  = df['avg_call_duration_min'].fillna(round(mean_call, 2))
    print(f"  ✅ Fix 5: Missing avg_call_duration_min filled ({round(mean_call,2)})")

    # ── Fix 6: Fill missing total_charges ────────────────────
    df['total_charges'] = df.apply(
        lambda x: round(x['recharge_amount'] * x['tenure_months'], 2)
        if pd.isnull(x['total_charges']) else x['total_charges'], axis=1
    )
    print("   ✅ Fix 6: Missing total_charges calculated from recharge × tenure")

    # ── Fix 7: Fill missing churn with mode ──────────────────
    churn_mode   = df['churn'].mode()[0]
    df['churn']  = df['churn'].fillna(churn_mode)
    print(f"  ✅ Fix 7: Missing churn filled with mode ({churn_mode})")

    # ── Fix 8: Remove duplicates ─────────────────────────────
    before_dedup = len(df)
    df           = df.drop_duplicates(subset=['customer_id'])
    print(f"  ✅ Fix 8: Removed {before_dedup - len(df)} duplicate records")

    # ── Fix 9: Remove internal ETL columns ───────────────────
    df = df.drop(columns=['_extracted_at', '_is_cleaned'], errors='ignore')

    # ── Fix 10: Add cleaned timestamp ────────────────────────
    df['_cleaned_at'] = datetime.now().isoformat()
    df['_is_cleaned'] = True

    print(f"\n   📊 Total records after cleaning : {len(df)}")
    print(f"   ✅ Null values after cleaning   :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    print("\n✅ Transformation complete!")
    return df

# ════════════════════════════════════════════════════════════════
#  STEP 4 — LOAD CLEANED DATA
#  Store cleaned data into MongoDB
# ════════════════════════════════════════════════════════════════
def load_cleaned(db, df):
    print("\n📦 STEP 4: LOADING cleaned data into MongoDB...")
    cleaned_collection = db['cleaned_customers']
    cleaned_collection.drop()

    records = df.to_dict(orient='records')
    cleaned_collection.insert_many(records)
    print(f"✅ {len(records)} cleaned records loaded into 'cleaned_customers' collection!")

# ════════════════════════════════════════════════════════════════
#  STEP 5 — LOG ETL RUN
# ════════════════════════════════════════════════════════════════
def log_etl(db, raw_count, cleaned_count):
    print("\n📝 STEP 5: LOGGING ETL run...")
    logs = db['etl_logs']
    log  = {
        "run_at":        datetime.now().isoformat(),
        "raw_records":   raw_count,
        "clean_records": cleaned_count,
        "dropped":       raw_count - cleaned_count,
        "status":        "success"
    }
    logs.insert_one(log)
    print("✅ ETL log saved!")

# ════════════════════════════════════════════════════════════════
#  MAIN — Run Full ETL Pipeline
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 Starting ETL Pipeline...")
    print("=" * 50)

    db           = get_db()
    customers    = extract()
    raw_data     = load_raw(db, customers)
    cleaned_df   = transform(customers)
    load_cleaned(db, cleaned_df)
    log_etl(db, len(customers), len(cleaned_df))

    print("\n" + "=" * 50)
    print("🎉 ETL Pipeline Complete!")
    print(f"   Raw records    : {len(customers)}")
    print(f"   Cleaned records: {len(cleaned_df)}")
    print("=" * 50)