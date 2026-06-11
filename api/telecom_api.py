from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

app = FastAPI(title="Telecom Dashboard API")

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── MongoDB ───────────────────────────────────────────────────
client = MongoClient("mongodb://localhost:27017/")
db = client["telecom_churn_db"]

# ══════════════════════════════════════════════════════════════
# BASIC ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "Telecom Dashboard API Live!"}


@app.get("/stats/overview")
def overview():
    total = db['customer_profiles'].count_documents({})
    churned = db['customer_profiles'].count_documents({"churn_predicted": "Yes"})
    urgent = db['customer_profiles'].count_documents({"offer.priority": "Urgent"})
    high_risk = db['customer_profiles'].count_documents({"risk_level": "High"})

    return {
        "total_customers": total,
        "predicted_churners": churned,
        "churn_rate": round(churned / total * 100, 2) if total else 0,
        "urgent_cases": urgent,
        "high_risk": high_risk
    }


@app.get("/stats/operator")
def operator_stats():
    pipeline = [
        {"$group": {
            "_id": "$operator",
            "total": {"$sum": 1},
            "churned": {"$sum": {"$cond": [{"$eq": ["$churn_predicted", "Yes"]}, 1, 0]}},
            "avg_risk": {"$avg": "$churn_risk_score"}
        }}
    ]
    return {"operators": list(db['customer_profiles'].aggregate(pipeline))}


@app.get("/stats/segments")
def segment_stats():
    pipeline = [{"$group": {"_id": "$customer_segment", "count": {"$sum": 1}}}]
    return {"segments": list(db['customer_profiles'].aggregate(pipeline))}


@app.get("/stats/rfm")
def rfm_stats():
    pipeline = [
        {"$group": {
            "_id": "$rfm_segment",
            "count": {"$sum": 1},
            "avg_churn_prob": {"$avg": "$churn_probability"}
        }}
    ]
    return {"rfm_segments": list(db['customer_profiles'].aggregate(pipeline))}


@app.get("/stats/offers")
def offer_stats():
    pipeline = [
        {"$group": {
            "_id": "$offer.offer_type",
            "count": {"$sum": 1}
        }}
    ]
    return {"offers": list(db['customer_profiles'].aggregate(pipeline))}


@app.get("/stats/loyalty")
def loyalty_stats():
    pipeline = [{"$group": {"_id": "$loyalty_tier", "count": {"$sum": 1}}}]
    return {"loyalty_tiers": list(db['customer_profiles'].aggregate(pipeline))}


@app.get("/stats/cities")
def city_stats():
    pipeline = [
        {"$group": {
            "_id": "$city",
            "total": {"$sum": 1},
            "churned": {"$sum": {"$cond": [{"$eq": ["$churn_predicted", "Yes"]}, 1, 0]}},
            "state": {"$first": "$state"}
        }},
        {"$sort": {"churned": -1}},
        {"$limit": 10}
    ]
    return {"top_cities": list(db['customer_profiles'].aggregate(pipeline))}


@app.get("/customers")
def get_customers():
    data = list(db['customer_profiles'].find({}, {'_id': 0}).limit(100))
    return {"customers": data}


# ══════════════════════════════════════════════════════════════
# FORECAST FIXED LOGIC
# ══════════════════════════════════════════════════════════════

def get_operator_forecasts():
    """
    Handles ALL formats:
    1. { operators: { Jio: {...} } }
    2. { forecasts: { Jio: {...} } }
    3. List of documents
    """

    doc = db['churn_forecasts'].find_one({}, {'_id': 0})

    if not doc:
        return []

    # 🔥 KEY FIX: support both keys
    data = doc.get("operators") or doc.get("forecasts") or {}

    if isinstance(data, dict):
        return list(data.values())

    if isinstance(data, list):
        return data

    return []


# ── Operator Summary ──────────────────────────────────────────
@app.get("/forecast/operators")
def forecast_operators():
    operators = get_operator_forecasts()

    forecasts = []
    for op in operators:
        forecasts.append({
            "operator": op.get("operator"),
            "current_customers": op.get("current_customers"),
            "current_churn_rate": op.get("current_rate"),
            "future_churn_rate": op.get("future_rate"),
            "total_6m_churners": op.get("total_predicted"),
            "avg_monthly": op.get("avg_per_month"),
            "trend": op.get("trend"),
            "scope": op.get("scope"),
            "scope_message": op.get("scope_message"),
        })

    return {"forecasts": forecasts}


# ── Monthly Forecast ──────────────────────────────────────────
@app.get("/forecast/monthly")
def forecast_monthly():
    operators = get_operator_forecasts()

    result = []
    for op in operators:
        monthly = []

        for m in op.get("forecast_6months", []):
            monthly.append({
                "month": m.get("month"),
                "churners": m.get("predicted"),
                "lower": m.get("lower"),
                "upper": m.get("upper")
            })

        result.append({
            "operator": op.get("operator"),
            "monthly": monthly
        })

    return {"monthly": result}


# ── Combined Forecast (BEST API) ──────────────────────────────
@app.get("/stats/forecast")
def forecast_stats():
    operators = get_operator_forecasts()

    forecasts = []
    for op in operators:
        monthly = [
            {
                "month": m.get("month"),
                "churners": m.get("predicted"),
                "lower": m.get("lower"),
                "upper": m.get("upper")
            }
            for m in op.get("forecast_6months", [])
        ]

        forecasts.append({
            "operator": op.get("operator"),
            "current_customers": op.get("current_customers"),
            "current_churn_rate": op.get("current_rate"),
            "future_churn_rate": op.get("future_rate"),
            "total_6m_churners": op.get("total_predicted"),
            "avg_monthly": op.get("avg_per_month"),
            "trend": op.get("trend"),
            "scope": op.get("scope"),
            "scope_message": op.get("scope_message"),
            "monthly_forecast": monthly
        })

    return {"forecasts": forecasts}

