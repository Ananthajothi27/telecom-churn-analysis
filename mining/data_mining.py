import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── MongoDB Connection ─────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "telecom_churn_db"

def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

# ── Load featured data ─────────────────────────────────────────
def load_data(db):
    print("📥 Loading featured data from MongoDB...")
    records = list(db['featured_customers'].find({}, {'_id': 0}))
    df      = pd.DataFrame(records)
    print(f"✅ Loaded {len(df)} records!")
    return df

# ════════════════════════════════════════════════════════════════
#  ALGORITHM 1 — APRIORI
# ════════════════════════════════════════════════════════════════
def run_apriori(db, df):
    print("\n🔍 ALGORITHM 1: Running Apriori...")

    transactions = []
    for _, row in df.iterrows():
        transaction = []

        transaction.append(f"Operator={row['operator']}")
        transaction.append(f"Plan={row['plan_type']}")
        transaction.append(f"Network={row['network_type']}")
        transaction.append(f"Loyalty={row['loyalty_tier']}")
        transaction.append(f"City={row['city_tier']}")
        transaction.append(f"DataUsage={row['data_usage_category']}")
        transaction.append(f"Tenure={row['tenure_category']}")

        if row['complaint_count'] == 0:
            transaction.append("Complaints=None")
        elif row['complaint_count'] <= 2:
            transaction.append("Complaints=Low")
        else:
            transaction.append("Complaints=High")

        if row['payment_delay_days'] == 0:
            transaction.append("PaymentDelay=None")
        elif row['payment_delay_days'] <= 5:
            transaction.append("PaymentDelay=Low")
        else:
            transaction.append("PaymentDelay=High")

        if row['churn_risk_score'] >= 0.6:
            transaction.append("RiskLevel=High")
        elif row['churn_risk_score'] >= 0.3:
            transaction.append("RiskLevel=Medium")
        else:
            transaction.append("RiskLevel=Low")

        transaction.append(f"Churn={row['churn']}")
        transactions.append(transaction)

    # Encode
    te     = TransactionEncoder()
    te_arr = te.fit(transactions).transform(transactions)
    df_te  = pd.DataFrame(te_arr, columns=te.columns_)

    # Run Apriori with lower support
    frequent_itemsets = apriori(
        df_te,
        min_support=0.05,
        use_colnames=True
    )

    # Generate rules with lower threshold
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.3,
        num_itemsets=len(frequent_itemsets)
    )

    # Filter churn rules
    churn_rules = rules[
        rules['consequents'].astype(str).str.contains('Churn=Yes')
    ].sort_values('confidence', ascending=False)

    print(f"✅ Found {len(frequent_itemsets)} frequent itemsets")
    print(f"✅ Found {len(rules)} association rules")
    print(f"✅ Found {len(churn_rules)} churn-related rules")

    if len(churn_rules) > 0:
        print("\n📊 Top 5 Churn Patterns:")
        for _, rule in churn_rules.head(5).iterrows():
            print(f"   IF {set(rule['antecedents'])} → THEN Churn")
            print(f"   Confidence: {rule['confidence']:.2f} | Support: {rule['support']:.2f} | Lift: {rule['lift']:.2f}")

    # Save to MongoDB
    rules_data = []
    for _, rule in churn_rules.head(20).iterrows():
        rules_data.append({
            "antecedents": list(rule['antecedents']),
            "consequents": list(rule['consequents']),
            "support":     round(rule['support'], 4),
            "confidence":  round(rule['confidence'], 4),
            "lift":        round(rule['lift'], 4),
            "created_at":  datetime.now().isoformat()
        })

    db['apriori_rules'].drop()
    if rules_data:
        db['apriori_rules'].insert_many(rules_data)
        print(f"✅ {len(rules_data)} churn rules saved to MongoDB!")
    else:
        print("⚠️ No churn rules found — all rules saved instead")
        all_rules = []
        for _, rule in rules.head(20).iterrows():
            all_rules.append({
                "antecedents": list(rule['antecedents']),
                "consequents": list(rule['consequents']),
                "support":     round(rule['support'], 4),
                "confidence":  round(rule['confidence'], 4),
                "lift":        round(rule['lift'], 4),
                "created_at":  datetime.now().isoformat()
            })
        db['apriori_rules'].insert_many(all_rules)
        print(f"✅ {len(all_rules)} general rules saved to MongoDB!")

# ════════════════════════════════════════════════════════════════
#  ALGORITHM 2 — K-MEANS CLUSTERING
# ════════════════════════════════════════════════════════════════
def run_kmeans(db, df):
    print("\n🔍 ALGORITHM 2: Running K-Means Clustering...")

    features = [
        'tenure_months',
        'monthly_charges',
        'total_charges',
        'complaint_count',
        'calls_per_day',
        'data_usage_gb',
        'churn_risk_score',
        'rfm_score'
    ]

    df_cluster = df[features].copy()
    scaler     = StandardScaler()
    df_scaled  = scaler.fit_transform(df_cluster)

    kmeans       = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df_scaled)

    cluster_summary = df.groupby('cluster').agg({
        'churn_risk_score': 'mean',
        'rfm_score':        'mean',
        'monthly_charges':  'mean',
        'tenure_months':    'mean',
        'churn':            lambda x: (x == 'Yes').mean()
    }).round(2)

    print("\n📊 Cluster Summary:")
    print(cluster_summary.to_string())

    cluster_labels = {}
    for cluster_id in range(4):
        row = cluster_summary.loc[cluster_id]
        if row['churn'] > 0.4:
            cluster_labels[cluster_id] = "High Churn Risk"
        elif row['rfm_score'] >= 9 and row['churn'] < 0.2:
            cluster_labels[cluster_id] = "Champions"
        elif row['tenure_months'] > 36:
            cluster_labels[cluster_id] = "Loyal Customers"
        else:
            cluster_labels[cluster_id] = "At Risk"

    df['cluster_label'] = df['cluster'].map(cluster_labels)

    print(f"\n✅ Cluster Labels:")
    for cluster_id, label in cluster_labels.items():
        count = len(df[df['cluster'] == cluster_id])
        churn_rate = cluster_summary.loc[cluster_id, 'churn']
        print(f"   Cluster {cluster_id} → {label}: {count} customers (churn rate: {churn_rate:.0%})")

    db['kmeans_clusters'].drop()
    records = df[['customer_id', 'cluster', 'cluster_label']].to_dict(orient='records')
    db['kmeans_clusters'].insert_many(records)
    print(f"✅ Cluster assignments saved to MongoDB!")

    return df

# ════════════════════════════════════════════════════════════════
#  ALGORITHM 3 — RFM ANALYSIS
# ════════════════════════════════════════════════════════════════
def run_rfm(db, df):
    print("\n🔍 ALGORITHM 3: Running RFM Analysis...")

    def rfm_segment(row):
        score = row['rfm_score']
        if score >= 11:   return "Champions"
        elif score >= 9:  return "Loyal Customers"
        elif score >= 7:  return "Potential Loyalists"
        elif score >= 5:  return "At Risk"
        else:             return "Lost Customers"

    df['rfm_segment'] = df.apply(rfm_segment, axis=1)

    print("✅ RFM Segments:")
    print(df['rfm_segment'].value_counts().to_string())

    print("\n📊 Churn Rate by RFM Segment:")
    rfm_churn = df.groupby('rfm_segment')['churn'].apply(
        lambda x: (x == 'Yes').mean()
    ).round(2)
    print(rfm_churn.to_string())

    db['rfm_analysis'].drop()
    rfm_records = df[[
        'customer_id', 'rfm_recency', 'rfm_frequency',
        'rfm_monetary', 'rfm_score', 'rfm_segment'
    ]].to_dict(orient='records')
    db['rfm_analysis'].insert_many(rfm_records)
    print(f"\n✅ RFM analysis saved to MongoDB!")

    return df

# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 Starting Data Mining...")
    print("=" * 50)

    db = get_db()
    df = load_data(db)

    run_apriori(db, df)
    df = run_kmeans(db, df)
    df = run_rfm(db, df)

    print("\n" + "=" * 50)
    print("🎉 Data Mining Complete!")
    print("   ✅ Apriori  → Churn patterns found")
    print("   ✅ K-Means  → Customers clustered")
    print("   ✅ RFM      → Customer value scored")
    print("=" * 50)