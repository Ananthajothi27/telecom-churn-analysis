import subprocess
import sys
import os
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime

# ── Path Setup ─────────────────────────────────────────────────
BASE_DIR   = "E:\\telecom-churn-analysis"
PYTHON     = sys.executable
FETCH      = os.path.join(BASE_DIR, "etl", "fetch_telecom_data.py")   # ✅ fetch fresh API data
ETL        = os.path.join(BASE_DIR, "etl", "etl_pipeline.py")
FEATURES   = os.path.join(BASE_DIR, "features", "feature_engineering.py")
MINING     = os.path.join(BASE_DIR, "mining", "data_mining.py")
ML         = os.path.join(BASE_DIR, "models", "ml_model.py")
PROFILING  = os.path.join(BASE_DIR, "profiling", "customer_profiling.py")
POWERBI    = os.path.join(BASE_DIR, "export_for_powerbi.py")           # ✅ export to Power BI

# ── Run Each Script ────────────────────────────────────────────
def run_script(script_path, name):
    print(f"\n>> Running {name}...")
    result = subprocess.run(
        [PYTHON, script_path],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}
    )
    if result.returncode == 0:
        print(f"[OK] {name} completed successfully!")
    else:
        print(f"[FAILED] {name} failed!")
        print(result.stderr)

# ── Full Pipeline ──────────────────────────────────────────────
def run_full_pipeline():
    print("\n" + "=" * 50)
    print(f"[START] Auto Pipeline Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    run_script(FETCH,     "Telecom Data Fetch (data.gov.in)")  # 1. Fetch fresh API data
    run_script(ETL,       "ETL Pipeline")                      # 2. Extract Transform Load
    run_script(FEATURES,  "Feature Engineering")               # 3. Build features
    run_script(MINING,    "Data Mining")                       # 4. Mine patterns
    run_script(ML,        "ML Model")                          # 5. Predict churn
    run_script(PROFILING, "Customer Profiling")                # 6. Profile customers
    run_script(POWERBI,   "Export for Power BI")               # 7. ✅ Export CSVs for Power BI

    print("\n" + "=" * 50)
    print(f"[DONE] Full Pipeline Complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

# ── Scheduler ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting pipeline once at startup...")
    run_full_pipeline()

    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_full_pipeline,
        'interval',
        hours=24,
        id='pipeline_job'
    )

    print("\n[SCHEDULER] Started!")
    print("   Pipeline will auto-run every 24 hours")
    print("   Press Ctrl+C to stop\n")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("\n[STOPPED] Scheduler stopped!")