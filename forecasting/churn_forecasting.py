import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# ── MongoDB Connection ─────────────────────────────────────────
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME   = "telecom_churn_db"

def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

# ════════════════════════════════════════════════════════════════
#  GENERATE HISTORICAL DATA PER OPERATOR
# ════════════════════════════════════════════════════════════════
def generate_operator_history(db):
    print("Loading data from MongoDB...")

    profiles = pd.DataFrame(list(db['customer_profiles'].find({}, {'_id': 0})))
    total    = len(profiles)

    print(f"Total customers : {total}")
    print(f"Operators found : {list(profiles['operator'].unique())}")

    n_periods = 12
    # FIX: Use list() to ensure dates is a plain Python list of Timestamps
    dates = list(pd.date_range(end=datetime.now(), periods=n_periods, freq='MS'))
    np.random.seed(42)

    operator_data = {}

    for operator in ['Jio', 'Airtel', 'Vi', 'BSNL']:
        op_df      = profiles[profiles['operator'] == operator]
        op_total   = len(op_df)
        op_churned = len(op_df[op_df['churn_predicted'] == 'Yes'])
        op_rate    = round(op_churned / max(1, op_total) * 100, 2)
        op_base    = max(1, op_churned)

        # FIX: Build monthly list using enumerate over dates to guarantee same length
        monthly = []
        for i, _ in enumerate(dates):
            variation = np.random.normal(0, max(1, op_base * 0.05))
            if operator in ['Vi', 'BSNL']:
                trend = i * 1.2
            else:
                trend = i * (-0.3)
            month_val = max(0, int(op_base + variation + trend))
            monthly.append(month_val)

        # FIX: Both arrays are now guaranteed to be length n_periods
        assert len(dates) == len(monthly), (
            f"Length mismatch for {operator}: dates={len(dates)}, monthly={len(monthly)}"
        )

        history_df = pd.DataFrame({
            'ds': dates,   # plain list of Timestamps — no DatetimeIndex wrapping needed
            'y':  monthly
        })

        operator_data[operator] = {
            'total':      op_total,
            'churned':    op_churned,
            'churn_rate': op_rate,
            'history':    history_df
        }

        print(f"  {operator}: {op_total} customers | "
              f"{op_churned} churned | "
              f"Rate: {op_rate}%")

    return operator_data, profiles

# ════════════════════════════════════════════════════════════════
#  PROPHET FORECAST PER OPERATOR
# ════════════════════════════════════════════════════════════════
def forecast_operators(operator_data, periods=6):
    print("\nRunning Prophet Forecast for each operator...")

    results = {}

    for operator, data in operator_data.items():
        print(f"\n  Forecasting {operator}...")
        try:
            m = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive'
            )
            m.fit(data['history'])

            future      = m.make_future_dataframe(periods=periods, freq='MS')
            forecast    = m.predict(future)
            future_only = forecast.tail(periods)

            monthly_forecast = []
            for _, row in future_only.iterrows():
                monthly_forecast.append({
                    "month":     row['ds'].strftime("%b %Y"),
                    "predicted": max(0, int(row['yhat'])),
                    "lower":     max(0, int(row['yhat_lower'])),
                    "upper":     max(0, int(row['yhat_upper']))
                })

            total_predicted = sum([m['predicted'] for m in monthly_forecast])
            avg_predicted   = round(total_predicted / periods, 1)
            future_rate     = round(
                total_predicted / max(1, data['total']) / periods * 100, 2
            )

            # Scope determination
            if data['churn_rate'] <= 20:
                scope     = "HIGH SCOPE"
                scope_msg = "Strong growth potential!"
            elif data['churn_rate'] <= 35:
                scope     = "MEDIUM SCOPE"
                scope_msg = "Moderate retention needed"
            else:
                scope     = "LOW SCOPE"
                scope_msg = "Urgent retention needed!"

            # Trend determination
            first_half  = sum([m['predicted'] for m in monthly_forecast[:3]])
            second_half = sum([m['predicted'] for m in monthly_forecast[3:]])
            trend = "Increasing" if second_half > first_half else "Decreasing"

            results[operator] = {
                "operator":          operator,
                "current_customers": data['total'],
                "current_churned":   data['churned'],
                "current_rate":      data['churn_rate'],
                "forecast_6months":  monthly_forecast,
                "total_predicted":   total_predicted,
                "avg_per_month":     avg_predicted,
                "future_rate":       future_rate,
                "trend":             trend,
                "scope":             scope,
                "scope_message":     scope_msg
            }

            print(f"    Current customers  : {data['total']}")
            print(f"    Current churned    : {data['churned']}")
            print(f"    Current churn rate : {data['churn_rate']}%")
            print(f"    Predicted 6 months : {total_predicted} churners")
            print(f"    Avg per month      : {avg_predicted}")
            print(f"    Future churn rate  : {future_rate}%")
            print(f"    Trend              : {trend}")
            print(f"    Scope              : {scope} - {scope_msg}")

        except Exception as e:
            print(f"    Prophet failed for {operator}: {e}")
            results[operator] = {
                "operator":          operator,
                "current_customers": data['total'],
                "current_churned":   data['churned'],
                "current_rate":      data['churn_rate'],
                "forecast_6months":  [],
                "total_predicted":   0,
                "avg_per_month":     0,
                "future_rate":       0,
                "trend":             "Unknown",
                "scope":             "Unknown",
                "scope_message":     "Forecast failed"
            }

    return results

# ════════════════════════════════════════════════════════════════
#  IDENTIFY HIGH SCOPE & LOW SCOPE
# ════════════════════════════════════════════════════════════════
def identify_scope(results):
    print("\n" + "=" * 55)
    print("OPERATOR FUTURE SCOPE ANALYSIS — 6 MONTHS")
    print("=" * 55)

    sorted_ops = sorted(
        results.items(),
        key=lambda x: x[1]['future_rate']
    )

    print("\n--- HIGHEST FUTURE SCOPE (Low Churn) ---")
    for op, data in sorted_ops[:2]:
        print(f"\n  {op}:")
        print(f"    Current Customers  : {data['current_customers']}")
        print(f"    Current Churn Rate : {data['current_rate']}%")
        print(f"    Future Churn Rate  : {data['future_rate']}%")
        print(f"    6-Month Churners   : {data['total_predicted']}")
        print(f"    Trend              : {data['trend']}")
        print(f"    Scope              : HIGH SCOPE")
        print(f"    Action             : Retain & Grow!")

    print("\n--- LOWEST FUTURE SCOPE (High Churn) ---")
    for op, data in sorted_ops[2:]:
        print(f"\n  {op}:")
        print(f"    Current Customers  : {data['current_customers']}")
        print(f"    Current Churn Rate : {data['current_rate']}%")
        print(f"    Future Churn Rate  : {data['future_rate']}%")
        print(f"    6-Month Churners   : {data['total_predicted']}")
        print(f"    Trend              : {data['trend']}")
        print(f"    Scope              : LOW SCOPE")
        print(f"    Action             : Urgent Retention Needed!")

    print("\n--- MONTHLY FORECAST PER OPERATOR ---")
    for op, data in results.items():
        print(f"\n  {op} ({data['scope']}):")
        if data['forecast_6months']:
            for m in data['forecast_6months']:
                print(f"    {m['month']}: {m['predicted']} churners "
                      f"[{m['lower']} - {m['upper']}]")

# ════════════════════════════════════════════════════════════════
#  SAVE TO MONGODB
# ════════════════════════════════════════════════════════════════
def save_to_mongodb(db, results):
    print("\nSaving forecast to MongoDB...")

    forecast_doc = {
        "forecast_date":   datetime.now().isoformat(),
        "forecast_period": "6 months",
        "algorithm":       "Prophet",
        "operators":       results
    }

    db['churn_forecasts'].drop()
    db['churn_forecasts'].insert_one(forecast_doc)
    print("Forecast saved to churn_forecasts collection!")

# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Starting Operator Future Scope Analysis...")
    print("=" * 55)

    db                      = get_db()
    operator_data, profiles = generate_operator_history(db)
    results                 = forecast_operators(operator_data)
    identify_scope(results)
    save_to_mongodb(db, results)

    print("\n" + "=" * 55)
    print("Forecasting Complete!")
    print("  Algorithm  : Prophet")
    print("  Focus      : Operator future scope + churn rate")
    print("  Saved      : MongoDB churn_forecasts collection")
    print("=" * 55)