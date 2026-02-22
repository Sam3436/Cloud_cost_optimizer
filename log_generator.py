import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker()

MICROSERVICES = [
    "auth-service", "payment-gateway", "recommendation-engine",
    "data-pipeline", "ml-training-job", "etl-processor",
    "api-gateway", "notification-service", "analytics-engine", "report-generator"
]

REGIONS = {
    "us-east-1": {"carbon_intensity_base": 380, "renewable_peak_hours": [10, 11, 14, 15]},
    "us-west-2": {"carbon_intensity_base": 120, "renewable_peak_hours": [11, 12, 13, 14, 15]},
    "eu-west-1": {"carbon_intensity_base": 200, "renewable_peak_hours": [9, 10, 11, 12, 13]},
    "ap-southeast-1": {"carbon_intensity_base": 450, "renewable_peak_hours": [13, 14]},
}

def generate_carbon_intensity(hour, region):
    """Simulate carbon intensity (gCO2/kWh) varying by hour and region."""
    base = REGIONS[region]["carbon_intensity_base"]
    peak_hours = REGIONS[region]["renewable_peak_hours"]
    # Lower carbon intensity during renewable-heavy hours (solar peak)
    reduction = 0.4 if hour in peak_hours else 0
    noise = np.random.normal(0, 20)
    return max(50, base * (1 - reduction) + noise)

def generate_logs(days=30, records_per_day=500):
    records = []
    start_date = datetime.now() - timedelta(days=days)

    for day in range(days):
        for _ in range(records_per_day):
            service = random.choice(MICROSERVICES)
            region = random.choice(list(REGIONS.keys()))
            timestamp = start_date + timedelta(
                days=day,
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            hour = timestamp.hour
            cpu_util = np.clip(np.random.beta(2, 5) * 100, 1, 99)
            mem_util = np.clip(np.random.beta(3, 4) * 100, 5, 99)
            duration_ms = np.random.lognormal(mean=6, sigma=1.5)
            invocations = random.randint(1, 500)
            # Power draw in kWh (simplified: CPU + memory contribution)
            power_kwh = (cpu_util / 100 * 0.005 + mem_util / 100 * 0.002) * (duration_ms / 3_600_000)
            carbon_intensity = generate_carbon_intensity(hour, region)
            carbon_gco2 = power_kwh * carbon_intensity
            cost_usd = power_kwh * 0.12 + (duration_ms / 1000) * 0.000001 * invocations

            records.append({
                "timestamp": timestamp,
                "service": service,
                "region": region,
                "hour_of_day": hour,
                "day_of_week": timestamp.weekday(),
                "cpu_utilization": round(cpu_util, 2),
                "memory_utilization": round(mem_util, 2),
                "duration_ms": round(duration_ms, 2),
                "invocations": invocations,
                "power_kwh": round(power_kwh, 8),
                "carbon_intensity_gco2_kwh": round(carbon_intensity, 2),
                "carbon_gco2": round(carbon_gco2, 6),
                "cost_usd": round(cost_usd, 6),
            })

    df = pd.DataFrame(records)
    df.to_csv("data/raw/cloudwatch_logs.csv", index=False)
    print(f" Generated {len(df)} log records → data/raw/cloudwatch_logs.csv")
    return df

if __name__ == "__main__":
    generate_logs()