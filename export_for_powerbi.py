import pandas as pd
from pymongo import MongoClient
import os

# ── MongoDB Connection ─────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "telecom_churn_db"

client = MongoClient(MONGO_URI)
db     = client[DB_NAME]

# ── Output folder ──────────────────────────────────────────────
OUTPUT_DIR = "E:\\telecom-churn-analysis\\data\\powerbi"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════
#  EXPORT ALL COLLECTIONS TO CSV
# ════════════════════════════════════════════════════════════════

# 1. Customer Profiles
print("Exporting customer profiles...")
profiles = pd.DataFrame(list(db['customer_profiles'].find({}, {'_id': 0})))
profiles['offer_type']     = profiles['offer'].apply(lambda x: x.get('offer_type', '') if isinstance(x, dict) else '')
profiles['offer_discount'] = profiles['offer'].apply(lambda x: x.get('discount', '') if isinstance(x, dict) else '')
profiles['offer_priority'] = profiles['offer'].apply(lambda x: x.get('priority', '') if isinstance(x, dict) else '')
profiles['offer_message']  = profiles['offer'].apply(lambda x: x.get('message', '') if isinstance(x, dict) else '')
profiles = profiles.drop(columns=['offer'], errors='ignore')
profiles.to_csv(f"{OUTPUT_DIR}\\customer_profiles.csv", index=False)
print(f"  Exported {len(profiles)} customer profiles!")

# 2. Churn Predictions
print("Exporting churn predictions...")
predictions = pd.DataFrame(list(db['churn_predictions'].find({}, {'_id': 0})))
predictions.to_csv(f"{OUTPUT_DIR}\\churn_predictions.csv", index=False)
print(f"  Exported {len(predictions)} predictions!")

# 3. RFM Analysis
print("Exporting RFM analysis...")
rfm = pd.DataFrame(list(db['rfm_analysis'].find({}, {'_id': 0})))
rfm.to_csv(f"{OUTPUT_DIR}\\rfm_analysis.csv", index=False)
print(f"  Exported {len(rfm)} RFM records!")

# 4. KMeans Clusters
print("Exporting clusters...")
clusters = pd.DataFrame(list(db['kmeans_clusters'].find({}, {'_id': 0})))
clusters.to_csv(f"{OUTPUT_DIR}\\kmeans_clusters.csv", index=False)
print(f"  Exported {len(clusters)} cluster records!")

# 5. Apriori Rules
print("Exporting Apriori rules...")
rules = pd.DataFrame(list(db['apriori_rules'].find({}, {'_id': 0})))
rules['antecedents'] = rules['antecedents'].apply(str)
rules['consequents'] = rules['consequents'].apply(str)
rules.to_csv(f"{OUTPUT_DIR}\\apriori_rules.csv", index=False)
print(f"  Exported {len(rules)} Apriori rules!")

# 6. ETL Logs
print("Exporting ETL logs...")
logs = pd.DataFrame(list(db['etl_logs'].find({}, {'_id': 0})))
logs.to_csv(f"{OUTPUT_DIR}\\etl_logs.csv", index=False)
print(f"  Exported {len(logs)} ETL logs!")

# 7. Forecast Data
print("Exporting forecast data...")
forecast_raw = list(db['churn_forecasts'].find({}, {'_id': 0}))
if forecast_raw:
    forecast = forecast_raw[0]
    operators = forecast.get('operators', {})

    forecast_rows = []
    for op, data in operators.items():
        for month_data in data.get('forecast_6months', []):
            forecast_rows.append({
                'operator':        op,
                'month':           month_data['month'],
                'predicted':       month_data['predicted'],
                'lower':           month_data['lower'],
                'upper':           month_data['upper'],
                'current_rate':    data['current_rate'],
                'future_rate':     data['future_rate'],
                'trend':           data['trend'],
                'scope':           data['scope']
            })

    if forecast_rows:
        forecast_df = pd.DataFrame(forecast_rows)
        forecast_df.to_csv(f"{OUTPUT_DIR}\\forecast_data.csv", index=False)
        print(f"  Exported {len(forecast_df)} forecast rows!")

print("\n" + "=" * 50)
print("All files exported to:")
print(f"  {OUTPUT_DIR}")
print("\nFiles created:")
for f in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(f"{OUTPUT_DIR}\\{f}")
    print(f"  {f} ({size} bytes)")
print("=" * 50)
print("Ready for Power BI!")