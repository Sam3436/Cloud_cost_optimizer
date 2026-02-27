import sys
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from codecarbon import EmissionsTracker

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Electricity Maps API ───────────────────────────────────────────────────────
def fetch_live_carbon_intensity(region_zone="US-NY-NYISO"):
    """
    Fetch real 24-hour carbon intensity from Electricity Maps API.
    Free tier available at: https://app.electricitymaps.com/map
    Sign up and add ELECTRICITY_MAPS_API_KEY to your .env file
    """
    api_key = os.getenv("YOUR_API_KEY")

    if not api_key:
        print("[WARNING] No ELECTRICITY_MAPS_API_KEY found in .env - using simulated data")
        return _fallback_carbon_intensity(region_zone)

    try:
        url = f"https://api.electricitymap.org/v3/carbon-intensity/history?zone={region_zone}"
        headers = {"auth-token": api_key}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        history = data.get("history", [])[-24:]
        if not history:
            print("[WARNING] Empty response from API - using simulated data")
            return _fallback_carbon_intensity(region_zone)

        intensities = [entry["carbonIntensity"] for entry in history]
        print(f"[OK] Fetched {len(intensities)} hours of live carbon data for {region_zone}")
        print(f"     Current intensity: {intensities[-1]} gCO2/kWh")
        print(f"     Min (cleanest): {min(intensities)} gCO2/kWh at hour {intensities.index(min(intensities))}")
        return intensities

    except requests.exceptions.ConnectionError:
        print("[ERROR] No internet connection - using simulated data")
        return _fallback_carbon_intensity(region_zone)
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] API error {e} - using simulated data")
        return _fallback_carbon_intensity(region_zone)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e} - using simulated data")
        return _fallback_carbon_intensity(region_zone)


def _fallback_carbon_intensity(region_zone):
    """Simulated profiles when API is unavailable."""
    profiles = {
        "US-NY-NYISO": [420,410,405,400,395,390,385,400,420,410,370,340,
                        320,310,330,360,390,410,430,440,445,440,435,430],
        "US-CAL-CISO": [180,175,170,165,160,158,155,148,130,110,90,80,
                        75,78,85,100,130,155,165,170,175,178,180,182],
        "IE":          [240,235,230,228,225,222,218,215,200,185,175,165,
                        160,162,170,185,200,220,235,245,248,245,242,240],
    }
    return profiles.get(region_zone, profiles["US-NY-NYISO"])


# ── 2. Save Live Data to CSV ──────────────────────────────────────────────────────
def save_live_intensity_to_csv(region_zone="US-NY-NYISO"):
    """Fetch live data and save it so the dashboard can read it."""
    intensities = fetch_live_carbon_intensity(region_zone)

    now = datetime.utcnow()
    records = []
    for i, ci in enumerate(intensities):
        records.append({
            "hour": i,
            "carbon_intensity_gco2_kwh": ci,
            "region_zone": region_zone,
            "fetched_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "is_live": os.getenv("ELECTRICITY_MAPS_API_KEY") is not None
        })

    df = pd.DataFrame(records)
    out_path = os.path.join(BASE_DIR, "data", "raw", "live_carbon_intensity.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved live carbon intensity -> {out_path}")
    return df


# ── 3. CodeCarbon Tracker ─────────────────────────────────────────────────────────
def track_carbon(func, *args, project_name="green-optimizer", **kwargs):
    """
    Wraps any function with CodeCarbon tracking.
    Usage: result, emissions = track_carbon(my_expensive_function, arg1, arg2)
    """
    output_dir = os.path.join(BASE_DIR, "data", "emissions")
    os.makedirs(output_dir, exist_ok=True)

    tracker = EmissionsTracker(
        project_name=project_name,
        output_dir=output_dir,
        log_level="error",        # suppress verbose logs
        save_to_file=True,
        save_to_logger=False
    )

    print(f"[TRACKING] Starting carbon tracking for: {project_name}")
    tracker.start()

    try:
        result = func(*args, **kwargs)
    finally:
        emissions_kg = tracker.stop()
        emissions_g = emissions_kg * 1000 if emissions_kg else 0
        print(f"[EMISSIONS] {project_name}: {emissions_g:.4f} gCO2eq ({emissions_kg:.6f} kg)")

    return result, emissions_g


# ── 4. Feed Live Data into the Pipeline ──────────────────────────────────────────
def run_live_pipeline(region_zone="US-NY-NYISO"):
    """
    Full pipeline run with live carbon data + emissions tracking.
    Call this instead of run_pipeline.py for live data integration.
    """
    print("\n" + "="*55)
    print("  GREEN CODING OPTIMIZER - LIVE PIPELINE")
    print("="*55)

    # Step 1: Fetch live carbon intensity
    print("\n[STEP 1] Fetching live carbon intensity data...")
    live_df = save_live_intensity_to_csv(region_zone)

    # Step 2: Generate/refresh logs with live intensity
    print("\n[STEP 2] Generating compute logs...")
    from log_generator import generate_logs
    _, logs_emissions = track_carbon(generate_logs, project_name="log-generation")

    # Step 3: Train model with emissions tracking
    print("\n[STEP 3] Training carbon prediction model...")
    from carbon_model import load_and_preprocess, train_model
    df = load_and_preprocess()
    _, model_emissions = track_carbon(train_model, df, project_name="model-training")

    # Step 4: Run optimizer with live carbon profile
    print("\n[STEP 4] Running LP optimizer with live carbon data...")
    from scheduler import optimize_batch_schedule

    live_intensities = live_df["carbon_intensity_gco2_kwh"].tolist()

    sample_jobs = [
        {"name": "ML Training",       "duration_hours": 3, "cpu_kwh": 0.5,  "memory_kwh": 0.2},
        {"name": "ETL Pipeline",      "duration_hours": 2, "cpu_kwh": 0.3,  "memory_kwh": 0.1},
        {"name": "Report Generation", "duration_hours": 1, "cpu_kwh": 0.1,  "memory_kwh": 0.05},
        {"name": "Data Backup",       "duration_hours": 2, "cpu_kwh": 0.2,  "memory_kwh": 0.08},
    ]

    result = optimize_batch_schedule(sample_jobs, region=region_zone)

    print(f"\n[RESULTS] Optimal Schedule:")
    for s in result["schedule"]:
        print(f"  {s['job']:<22} Start {s['start_hour']:02d}:00 -> {s['end_hour']:02d}:00 | {s['carbon_gco2']} gCO2")

    print(f"\n  Total Carbon (optimized): {result['total_carbon_gco2']} gCO2")
    print(f"  Total Carbon (naive):     {result['naive_carbon_gco2']} gCO2")
    print(f"  Carbon Saved:             {result['carbon_saved_gco2']} gCO2 ({result['savings_pct']}%)")

    # Step 5: Summary of pipeline's own emissions
    total_pipeline_emissions = logs_emissions + model_emissions
    print(f"\n[PIPELINE FOOTPRINT] This pipeline run consumed: {total_pipeline_emissions:.4f} gCO2eq")
    print("\n[DONE] Launch dashboard with: streamlit run app.py")
    print("="*55)


if __name__ == "__main__":
    # Change zone to match your region:
    # US-NY-NYISO  -> New York
    # US-CAL-CISO  -> California  
    # IE           -> Ireland
    # DE           -> Germany
    # GB           -> Great Britain
    run_live_pipeline(region_zone="US-NY-NYISO")
