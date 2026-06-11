import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime

# ── MongoDB Connection ─────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "telecom_churn_db"

def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

# ── Load cleaned data from MongoDB ────────────────────────────
def load_data(db):
    print("📥 Loading cleaned data from MongoDB...")
    records = list(db['cleaned_customers'].find({}, {'_id': 0}))
    df      = pd.DataFrame(records)
    print(f"✅ Loaded {len(df)} records!")
    return df

# ════════════════════════════════════════════════════════════════
#  FEATURE 1 — Loyalty Tier
#  Based on tenure months
# ════════════════════════════════════════════════════════════════
def add_loyalty_tier(df):
    def get_tier(tenure):
        if tenure <= 6:   return "Bronze"
        elif tenure <= 24: return "Silver"
        elif tenure <= 48: return "Gold"
        else:              return "Platinum"
    df['loyalty_tier'] = df['tenure_months'].apply(get_tier)
    print("✅ Feature 1: loyalty_tier added")
    return df

# ════════════════════════════════════════════════════════════════
#  FEATURE 2 — RFM Scores
#  Recency, Frequency, Monetary
# ════════════════════════════════════════════════════════════════
def add_rfm_scores(df):
    # Recency → lower last_recharge_days = better
    df['rfm_recency'] = pd.cut(
        df['last_recharge_days'],
        bins=[0, 7, 30, 60, 90],
        labels=[4, 3, 2, 1],
        include_lowest=True
    ).astype(int)

    # Frequency → higher calls_per_day = better
    df['rfm_frequency'] = pd.cut(
        df['calls_per_day'],
        bins=[0, 3, 6, 10, 15],
        labels=[1, 2, 3, 4],
        include_lowest=True
    ).astype(int)

    # Monetary → higher total_charges = better
    df['rfm_monetary'] = pd.cut(
        df['total_charges'],
        bins=[0, 5000, 15000, 30000, 100000],
        labels=[1, 2, 3, 4],
        include_lowest=True
    ).astype(int)

    # Combined RFM Score
    df['rfm_score'] = df['rfm_recency'] + df['rfm_frequency'] + df['rfm_monetary']
    print("✅ Feature 2: rfm_recency, rfm_frequency, rfm_monetary, rfm_score added")
    return df

# ════════════════════════════════════════════════════════════════
#  FEATURE 3 — Churn Risk Score
#  Higher score = higher churn risk
# ════════════════════════════════════════════════════════════════
def add_churn_risk_score(df):
    # Operator risk weight
    operator_risk = {"Vi": 0.45, "BSNL": 0.38, "Airtel": 0.20, "Jio": 0.15}
    df['operator_risk'] = df['operator'].map(operator_risk)

    # Normalize features
    df['churn_risk_score'] = (
        (df['complaint_count'] / 6) * 0.30 +
        (df['customer_service_calls'] / 8) * 0.20 +
        (df['payment_delay_days'] / 15) * 0.15 +
        (df['last_recharge_days'] / 90) * 0.20 +
        df['operator_risk'] * 0.15
    ).round(4)

    print("✅ Feature 3: churn_risk_score added")
    return df

# ════════════════════════════════════════════════════════════════
#  FEATURE 4 — Average Daily Spend
# ════════════════════════════════════════════════════════════════
def add_avg_daily_spend(df):
    df['avg_daily_spend'] = (df['monthly_charges'] / 30).round(2)
    print("✅ Feature 4: avg_daily_spend added")
    return df

# ════════════════════════════════════════════════════════════════
#  FEATURE 5 — Is High Value Customer
# ════════════════════════════════════════════════════════════════
def add_high_value_flag(df):
    df['is_high_value'] = (df['rfm_score'] >= 9).astype(int)
    print("✅ Feature 5: is_high_value flag added")
    return df

# ════════════════════════════════════════════════════════════════
#  FEATURE 6 — Customer Segment
#  Based on RFM score + Churn Risk
# ════════════════════════════════════════════════════════════════
def add_customer_segment(df):
    def get_segment(row):
        if row['churn_risk_score'] >= 0.6:
            return "High Risk"
        elif row['churn_risk_score'] >= 0.4:
            if row['rfm_score'] >= 9:
                return "Valuable At Risk"
            return "Medium Risk"
        else:
            if row['rfm_score'] >= 9:
                return "Champion"
            elif row['rfm_score'] >= 6:
                return "Loyal"
            return "Needs Attention"

    df['customer_segment'] = df.apply(get_segment, axis=1)
    print("✅ Feature 6: customer_segment added")
    return df

# ════════════════════════════════════════════════════════════════
#  FEATURE 7 — Data Usage Category
# ════════════════════════════════════════════════════════════════
def add_data_usage_category(df):
    df['data_usage_category'] = pd.cut(
        df['data_usage_gb'],
        bins=[0, 5, 15, 30, 100],
        labels=['Low', 'Medium', 'High', 'Very High'],
        include_lowest=True
    )
    print("✅ Feature 7: data_usage_category added")
    return df

# ════════════════════════════════════════════════════════════════
#  FEATURE 8 — Tenure Category
# ════════════════════════════════════════════════════════════════
def add_tenure_category(df):
    df['tenure_category'] = pd.cut(
        df['tenure_months'],
        bins=[0, 12, 24, 48, 72],
        labels=['New', 'Regular', 'Established', 'Veteran'],
        include_lowest=True
    )
    print("✅ Feature 8: tenure_category added")
    return df

# ════════════════════════════════════════════════════════════════
#  SAVE TO MONGODB
# ════════════════════════════════════════════════════════════════
def save_features(db, df):
    print("\n📦 Saving featured data to MongoDB...")
    collection = db['featured_customers']
    collection.drop()

    # Convert categorical columns to string
    cat_cols = df.select_dtypes(['category']).columns
    df[cat_cols] = df[cat_cols].astype(str)

    df['_featured_at'] = datetime.now().isoformat()
    records = df.to_dict(orient='records')
    collection.insert_many(records)
    print(f"✅ {len(records)} records saved to 'featured_customers' collection!")

# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 Starting Feature Engineering...")
    print("=" * 50)

    db = get_db()
    df = load_data(db)

    df = add_loyalty_tier(df)
    df = add_rfm_scores(df)
    df = add_churn_risk_score(df)
    df = add_avg_daily_spend(df)
    df = add_high_value_flag(df)
    df = add_customer_segment(df)
    df = add_data_usage_category(df)
    df = add_tenure_category(df)

    save_features(db, df)

    print("\n" + "=" * 50)
    print("🎉 Feature Engineering Complete!")
    print(f"   Total features created : 8")
    print(f"   Total records          : {len(df)}")
    print("\n📊 Customer Segments:")
    print(df['customer_segment'].value_counts())
    print("\n📊 Loyalty Tiers:")
    print(df['loyalty_tier'].value_counts())
    print("=" * 50)