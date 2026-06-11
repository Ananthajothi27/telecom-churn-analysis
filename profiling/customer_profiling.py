import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── MongoDB Connection ─────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "telecom_churn_db"

def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

# ════════════════════════════════════════════════════════════════
#  LOAD ALL COLLECTIONS
# ════════════════════════════════════════════════════════════════
def load_all_data(db):
    print("Loading all collections from MongoDB...")

    customers   = pd.DataFrame(list(db['featured_customers'].find({}, {'_id': 0})))
    predictions = pd.DataFrame(list(db['churn_predictions'].find({}, {'_id': 0})))
    rfm         = pd.DataFrame(list(db['rfm_analysis'].find({}, {'_id': 0})))
    clusters    = pd.DataFrame(list(db['kmeans_clusters'].find({}, {'_id': 0})))

    print(f"Loaded all collections!")
    print(f"Featured customers columns: {list(customers.columns)}")
    return customers, predictions, rfm, clusters

# ════════════════════════════════════════════════════════════════
#  MERGE ALL DATA
# ════════════════════════════════════════════════════════════════
def merge_data(customers, predictions, rfm, clusters):
    print("\nMerging all data...")

    df = customers.merge(
        predictions[['customer_id', 'churn_predicted',
                     'churn_probability', 'risk_level']],
        on='customer_id', how='left'
    )
    df = df.merge(
        rfm[['customer_id', 'rfm_segment']],
        on='customer_id', how='left'
    )
    df = df.merge(
        clusters[['customer_id', 'cluster_label']],
        on='customer_id', how='left'
    )

    print(f"Merged data: {len(df)} customers!")
    return df

# ════════════════════════════════════════════════════════════════
#  PERSONALIZED OFFER ENGINE
# ════════════════════════════════════════════════════════════════
def generate_offer(row):

    risk       = row.get('risk_level', 'Low')
    operator   = row.get('operator', 'Jio')
    rfm_seg    = row.get('rfm_segment', 'At Risk')
    loyalty    = row.get('loyalty_tier', 'Bronze')
    plan       = row.get('plan_type', 'Prepaid')
    churn_prob = float(row.get('churn_probability', 0))

    # ── High Risk Customers → Retention Offers ──────────────
    if risk == 'High' or churn_prob >= 0.6:
        if loyalty in ['Gold', 'Platinum']:
            return {
                "offer_type":   "Premium Retention",
                "discount":     "40%",
                "free_days":    30,
                "plan_upgrade": "Free 5G Upgrade",
                "message":      f"Exclusive: 40% off + Free 5G upgrade for our valued {loyalty} member!",
                "priority":     "Urgent"
            }
        elif operator == 'Vi':
            return {
                "offer_type":   "Vi Retention Special",
                "discount":     "35%",
                "free_days":    15,
                "plan_upgrade": "Free OTT Bundle",
                "message":      "Stay with Vi! Get 35% off + Free Netflix & Hotstar for 3 months!",
                "priority":     "Urgent"
            }
        else:
            return {
                "offer_type":   "Retention Offer",
                "discount":     "30%",
                "free_days":    14,
                "plan_upgrade": "Free Data Boost",
                "message":      "Special offer: 30% off your next recharge + 2GB extra daily!",
                "priority":     "High"
            }

    # ── Medium Risk → Engagement Offers ─────────────────────
    elif risk == 'Medium' or churn_prob >= 0.3:
        if rfm_seg == 'Champions' or rfm_seg == 'Loyal Customers':
            return {
                "offer_type":   "Loyalty Reward",
                "discount":     "20%",
                "free_days":    7,
                "plan_upgrade": "Free Weekend Data",
                "message":      "Thank you for your loyalty! Enjoy 20% off + free weekend data!",
                "priority":     "Medium"
            }
        elif plan == 'Prepaid':
            return {
                "offer_type":   "Prepaid Special",
                "discount":     "15%",
                "free_days":    5,
                "plan_upgrade": "Free SMS Pack",
                "message":      "Recharge now and get 15% cashback + Free SMS pack!",
                "priority":     "Medium"
            }
        else:
            return {
                "offer_type":   "Engagement Offer",
                "discount":     "10%",
                "free_days":    3,
                "plan_upgrade": "Bonus Data",
                "message":      "Exclusive offer: 10% off + 1GB bonus data daily for a week!",
                "priority":     "Medium"
            }

    # ── Low Risk → Upsell Offers ─────────────────────────────
    else:
        if rfm_seg == 'Champions':
            return {
                "offer_type":   "Premium Upsell",
                "discount":     "0%",
                "free_days":    0,
                "plan_upgrade": "5G Plan Upgrade",
                "message":      "You are our Champion! Upgrade to 5G and experience blazing speeds!",
                "priority":     "Low"
            }
        elif loyalty == 'Platinum':
            return {
                "offer_type":   "Platinum Reward",
                "discount":     "10%",
                "free_days":    7,
                "plan_upgrade": "Premium OTT Bundle",
                "message":      "Platinum member exclusive: Free OTT bundle + 10% off annual plan!",
                "priority":     "Low"
            }
        else:
            return {
                "offer_type":   "Standard Offer",
                "discount":     "5%",
                "free_days":    0,
                "plan_upgrade": "Data Booster",
                "message":      "Happy with our service? Refer a friend and get Rs.100 cashback!",
                "priority":     "Low"
            }

# ════════════════════════════════════════════════════════════════
#  BUILD CUSTOMER PROFILES
# ════════════════════════════════════════════════════════════════
def build_profiles(db, df):
    print("\nBuilding dynamic customer profiles...")

    # Check available columns
    available_cols = list(df.columns)
    print(f"Available columns: {available_cols}")

    profiles = []
    for _, row in df.iterrows():
        offer   = generate_offer(row)
        profile = {
            # Identity
            "customer_id":       row['customer_id'],
            "name":              row['name'],
            "age":               row['age'],
            "gender":            row['gender'],
            "state":             row['state'],
            "city":              row.get('city', 'Unknown'),
            "pincode":           row.get('pincode', '000000'),
            "city_tier":         row['city_tier'],

            # Service
            "operator":          row['operator'],
            "plan_type":         row['plan_type'],
            "network_type":      row['network_type'],
            "recharge_amount":   row['recharge_amount'],
            "tenure_months":     row['tenure_months'],

            # Usage
            "data_usage_gb":     row['data_usage_gb'],
            "calls_per_day":     row['calls_per_day'],
            "monthly_charges":   row['monthly_charges'],
            "total_charges":     row['total_charges'],

            # Engineered Features
            "loyalty_tier":      row['loyalty_tier'],
            "churn_risk_score":  row['churn_risk_score'],
            "rfm_score":         row['rfm_score'],
            "avg_daily_spend":   row['avg_daily_spend'],
            "is_high_value":     row['is_high_value'],
            "customer_segment":  row['customer_segment'],

            # ML Predictions
            "churn_predicted":   row.get('churn_predicted', 'No'),
            "churn_probability": float(row.get('churn_probability', 0)),
            "risk_level":        row.get('risk_level', 'Low'),

            # Mining Results
            "rfm_segment":       row.get('rfm_segment', 'Unknown'),
            "cluster_label":     row.get('cluster_label', 'Unknown'),

            # Personalized Offer
            "offer":             offer,

            # Metadata
            "profile_created":   datetime.now().isoformat(),
            "last_updated":      datetime.now().isoformat()
        }
        profiles.append(profile)

    db['customer_profiles'].drop()
    db['customer_profiles'].insert_many(profiles)
    print(f"{len(profiles)} customer profiles created!")
    return profiles

# ════════════════════════════════════════════════════════════════
#  OFFER SUMMARY
# ════════════════════════════════════════════════════════════════
def print_offer_summary(profiles):
    print("\nOffer Distribution:")
    offer_types = {}
    priorities  = {}

    for p in profiles:
        ot = p['offer']['offer_type']
        pr = p['offer']['priority']
        offer_types[ot] = offer_types.get(ot, 0) + 1
        priorities[pr]  = priorities.get(pr, 0) + 1

    print("\n   By Offer Type:")
    for k, v in sorted(offer_types.items(), key=lambda x: x[1], reverse=True):
        print(f"   {k}: {v} customers")

    print("\n   By Priority:")
    for k, v in sorted(priorities.items(), key=lambda x: x[1], reverse=True):
        print(f"   {k}: {v} customers")

# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Starting Customer Profiling & Offer Engine...")
    print("=" * 50)

    db                                    = get_db()
    customers, predictions, rfm, clusters = load_all_data(db)
    df                                    = merge_data(customers, predictions, rfm, clusters)
    profiles                              = build_profiles(db, df)
    print_offer_summary(profiles)

    print("\n" + "=" * 50)
    print("Customer Profiling Complete!")
    print(f"   Total profiles created : {len(profiles)}")
    print(f"   Collections updated    : customer_profiles")
    print("=" * 50)