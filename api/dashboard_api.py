from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

app = FastAPI(title="Telecom Dashboard API")

# ── Allow React to connect ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── MongoDB ────────────────────────────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db     = client["telecom_churn_db"]

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Dashboard API Live!"}

@app.get("/stats/overview")
def overview():
    total     = db['customer_profiles'].count_documents({})
    churned   = db['customer_profiles'].count_documents({"churn_predicted": "Yes"})
    urgent    = db['customer_profiles'].count_documents({"offer.priority": "Urgent"})
    high_risk = db['customer_profiles'].count_documents({"risk_level": "High"})
    return {
        "total_customers":    total,
        "predicted_churners": churned,
        "churn_rate":         round(churned/total*100, 2),
        "urgent_cases":       urgent,
        "high_risk":          high_risk
    }

@app.get("/stats/operator")
def operator_stats():
    pipeline = [
        {"$group": {
            "_id":      "$operator",
            "total":    {"$sum": 1},
            "churned":  {"$sum": {"$cond": [{"$eq": ["$churn_predicted", "Yes"]}, 1, 0]}},
            "avg_risk": {"$avg": "$churn_risk_score"}
        }}
    ]
    result = list(db['customer_profiles'].aggregate(pipeline))
    return {"operators": result}

@app.get("/stats/segments")
def segment_stats():
    pipeline = [
        {"$group": {
            "_id":   "$customer_segment",
            "count": {"$sum": 1}
        }}
    ]
    result = list(db['customer_profiles'].aggregate(pipeline))
    return {"segments": result}

@app.get("/stats/rfm")
def rfm_stats():
    pipeline = [
        {"$group": {
            "_id":            "$rfm_segment",
            "count":          {"$sum": 1},
            "avg_churn_prob": {"$avg": "$churn_probability"}
        }}
    ]
    result = list(db['customer_profiles'].aggregate(pipeline))
    return {"rfm_segments": result}

@app.get("/stats/offers")
def offer_stats():
    pipeline = [
        {"$group": {
            "_id":      "$offer.offer_type",
            "count":    {"$sum": 1},
            "priority": {"$first": "$offer.priority"}
        }}
    ]
    result = list(db['customer_profiles'].aggregate(pipeline))
    return {"offers": result}

@app.get("/stats/loyalty")
def loyalty_stats():
    pipeline = [
        {"$group": {
            "_id":            "$loyalty_tier",
            "count":          {"$sum": 1},
            "avg_churn_prob": {"$avg": "$churn_probability"}
        }}
    ]
    result = list(db['customer_profiles'].aggregate(pipeline))
    return {"loyalty_tiers": result}

# ── City Stats ─────────────────────────────────────────────────
@app.get("/stats/cities")
def city_stats():
    pipeline = [
        {"$group": {
            "_id":     "$city",
            "total":   {"$sum": 1},
            "churned": {"$sum": {"$cond": [{"$eq": ["$churn_predicted", "Yes"]}, 1, 0]}},
            "state":   {"$first": "$state"},
            "pincode": {"$first": "$pincode"}
        }},
        {"$sort":  {"churned": -1}},
        {"$limit": 10}
    ]
    result = list(db['customer_profiles'].aggregate(pipeline))
    return {"top_cities": result}

# ── State Stats ────────────────────────────────────────────────
@app.get("/stats/states")
def state_stats():
    pipeline = [
        {"$group": {
            "_id":      "$state",
            "total":    {"$sum": 1},
            "churned":  {"$sum": {"$cond": [{"$eq": ["$churn_predicted", "Yes"]}, 1, 0]}},
            "avg_risk": {"$avg": "$churn_risk_score"}
        }},
        {"$sort": {"churned": -1}},
        {"$limit": 10}
    ]
    result = list(db['customer_profiles'].aggregate(pipeline))
    return {"top_states": result}

# ── City Tier Stats ────────────────────────────────────────────
@app.get("/stats/city-tiers")
def city_tier_stats():
    pipeline = [
        {"$group": {
            "_id":            "$city_tier",
            "total":          {"$sum": 1},
            "churned":        {"$sum": {"$cond": [{"$eq": ["$churn_predicted", "Yes"]}, 1, 0]}},
            "avg_churn_prob": {"$avg": "$churn_probability"}
        }}
    ]
    result = list(db['customer_profiles'].aggregate(pipeline))
    return {"city_tiers": result}

# ── NEW: Forecast Stats ────────────────────────────────────────
@app.get("/stats/forecast")
def forecast_stats():
    records = list(db['churn_forecasts'].find({}, {'_id': 0}))

    if not records:
        return {"forecasts": [], "summary": []}

    # Build summary per operator
    summary = []
    for rec in records:
        summary.append({
            "operator":          rec.get("operator"),
            "current_customers": rec.get("current_customers"),
            "current_churned":   rec.get("current_churned"),
            "current_churn_rate":rec.get("current_churn_rate"),
            "future_churn_rate": rec.get("future_churn_rate"),
            "total_6m_churners": rec.get("total_6m_churners"),
            "avg_monthly":       rec.get("avg_monthly"),
            "trend":             rec.get("trend"),
            "scope":             rec.get("scope"),
            "action":            rec.get("action"),
            "monthly_forecast":  rec.get("monthly_forecast", [])
        })

    # Sort by future churn rate ascending (best first)
    summary.sort(key=lambda x: x.get("future_churn_rate") or 0)

    return {"forecasts": summary, "total_operators": len(summary)}

# ── Customers ──────────────────────────────────────────────────
@app.get("/customers")
def get_customers():
    records = list(db['customer_profiles'].find({}, {'_id': 0}).limit(100))
    return {"total": len(records), "customers": records}

@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    record = db['customer_profiles'].find_one(
        {"customer_id": customer_id}, {'_id': 0}
    )
    if record:
        return record
    return {"error": "Customer not found"}

@app.get("/apriori")
def get_apriori():
    rules = list(db['apriori_rules'].find({}, {'_id': 0}))
    return {"rules": rules}