import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ── MongoDB Connection ─────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "telecom_churn_db"

def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

# ── Load Data ──────────────────────────────────────────────────
def load_data(db):
    print("📥 Loading featured data from MongoDB...")
    records = list(db['featured_customers'].find({}, {'_id': 0}))
    df      = pd.DataFrame(records)
    print(f"✅ Loaded {len(df)} records!")
    return df

# ════════════════════════════════════════════════════════════════
#  STEP 1 — PREPARE DATA
# ════════════════════════════════════════════════════════════════
def prepare_data(df):
    print("\n🔄 STEP 1: Preparing data...")

    # Select features for ML
    feature_cols = [
        'tenure_months', 'monthly_charges', 'total_charges',
        'complaint_count', 'calls_per_day', 'data_usage_gb',
        'sms_per_day', 'payment_delay_days', 'customer_service_calls',
        'last_recharge_days', 'churn_risk_score', 'rfm_score',
        'avg_daily_spend', 'is_high_value', 'outstanding_amount',
        'operator', 'plan_type', 'network_type', 'city_tier',
        'loyalty_tier', 'data_usage_category', 'tenure_category',
        'roaming_usage', 'auto_pay', 'gender'
    ]

    df_ml = df[feature_cols + ['churn']].copy()

    # Encode categorical columns
    le      = LabelEncoder()
    cat_cols = [
        'operator', 'plan_type', 'network_type', 'city_tier',
        'loyalty_tier', 'data_usage_category', 'tenure_category',
        'roaming_usage', 'auto_pay', 'gender'
    ]

    for col in cat_cols:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

    # Target variable
    df_ml['churn'] = (df_ml['churn'] == 'Yes').astype(int)

    X = df_ml.drop('churn', axis=1)
    y = df_ml['churn']

    print(f"✅ Features prepared: {X.shape[1]} features")
    print(f"   Churn distribution before SMOTE:")
    print(f"   No Churn: {(y==0).sum()} | Churn: {(y==1).sum()}")

    return X, y

# ════════════════════════════════════════════════════════════════
#  STEP 2 — APPLY SMOTE
#  Fix imbalanced data
# ════════════════════════════════════════════════════════════════
def apply_smote(X, y):
    print("\n🔄 STEP 2: Applying SMOTE...")
    smote        = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"✅ After SMOTE:")
    print(f"   No Churn: {(y_resampled==0).sum()} | Churn: {(y_resampled==1).sum()}")
    return X_resampled, y_resampled

# ════════════════════════════════════════════════════════════════
#  STEP 3 — TRAIN RANDOM FOREST
# ════════════════════════════════════════════════════════════════
def train_model(X, y):
    print("\n🔄 STEP 3: Training Random Forest...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Random Forest trained!")

    return model, X_train, X_test, y_train, y_test

# ════════════════════════════════════════════════════════════════
#  STEP 4 — EVALUATE MODEL
# ════════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, X):
    print("\n🔄 STEP 4: Evaluating model...")

    y_pred     = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy   = accuracy_score(y_test, y_pred)
    roc_auc    = roc_auc_score(y_test, y_pred_prob)
    cm         = confusion_matrix(y_test, y_pred)
    report     = classification_report(y_test, y_pred)

    print(f"\n📊 Model Performance:")
    print(f"   Accuracy  : {accuracy:.2%}")
    print(f"   ROC AUC   : {roc_auc:.2%}")
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Negative  : {cm[0][0]}")
    print(f"   False Positive : {cm[0][1]}")
    print(f"   False Negative : {cm[1][0]}")
    print(f"   True Positive  : {cm[1][1]}")
    print(f"\n📊 Classification Report:")
    print(report)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature':   X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("📊 Top 10 Important Features:")
    print(feature_importance.head(10).to_string())

    return accuracy, roc_auc, feature_importance

# ════════════════════════════════════════════════════════════════
#  STEP 5 — SAVE PREDICTIONS TO MONGODB
# ════════════════════════════════════════════════════════════════
def save_predictions(db, df, model, X, y):
    print("\n🔄 STEP 5: Saving predictions to MongoDB...")

    # Prepare full dataset for prediction
    feature_cols = [
        'tenure_months', 'monthly_charges', 'total_charges',
        'complaint_count', 'calls_per_day', 'data_usage_gb',
        'sms_per_day', 'payment_delay_days', 'customer_service_calls',
        'last_recharge_days', 'churn_risk_score', 'rfm_score',
        'avg_daily_spend', 'is_high_value', 'outstanding_amount',
        'operator', 'plan_type', 'network_type', 'city_tier',
        'loyalty_tier', 'data_usage_category', 'tenure_category',
        'roaming_usage', 'auto_pay', 'gender'
    ]

    from sklearn.preprocessing import LabelEncoder
    le      = LabelEncoder()
    cat_cols = [
        'operator', 'plan_type', 'network_type', 'city_tier',
        'loyalty_tier', 'data_usage_category', 'tenure_category',
        'roaming_usage', 'auto_pay', 'gender'
    ]

    df_pred = df[feature_cols].copy()
    for col in cat_cols:
        df_pred[col] = le.fit_transform(df_pred[col].astype(str))

    # Predict churn probability for all customers
    churn_proba = model.predict_proba(df_pred)[:, 1]
    churn_pred  = model.predict(df_pred)

    # Save predictions
    predictions = []
    for i, row in df.iterrows():
        predictions.append({
            "customer_id":       row['customer_id'],
            "name":              row['name'],
            "operator":          row['operator'],
            "plan_type":         row['plan_type'],
            "churn_actual":      row['churn'],
            "churn_predicted":   "Yes" if churn_pred[i] == 1 else "No",
            "churn_probability": round(float(churn_proba[i]), 4),
            "risk_level":        "High" if churn_proba[i] >= 0.6 else "Medium" if churn_proba[i] >= 0.3 else "Low",
            "customer_segment":  row['customer_segment'],
            "rfm_segment":       row.get('rfm_segment', 'Unknown'),
            "predicted_at":      datetime.now().isoformat()
        })

    db['churn_predictions'].drop()
    db['churn_predictions'].insert_many(predictions)
    print(f"✅ {len(predictions)} predictions saved to MongoDB!")

# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 Starting ML Model (Random Forest + SMOTE)...")
    print("=" * 50)

    db  = get_db()
    df  = load_data(db)
    X, y = prepare_data(df)
    X_resampled, y_resampled = apply_smote(X, y)
    model, X_train, X_test, y_train, y_test = train_model(
        X_resampled, y_resampled
    )
    accuracy, roc_auc, feature_importance = evaluate_model(
        model, X_test, y_test, X
    )
    save_predictions(db, df, model, X, y)

    print("\n" + "=" * 50)
    print("🎉 ML Model Complete!")
    print(f"   Accuracy : {accuracy:.2%}")
    print(f"   ROC AUC  : {roc_auc:.2%}")
    print("=" * 50)