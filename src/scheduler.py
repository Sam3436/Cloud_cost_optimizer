import pandas as pd
import numpy as np
from pulp import (
    LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value, LpBinary
)

def get_hourly_carbon_intensity(region="us-east-1"):
    """Returns 24-hour carbon intensity profile for a region (gCO2/kWh)."""
    profiles = {
        "us-east-1": [420,410,405,400,395,390,385,400,420,410,370,340,
                      320,310,330,360,390,410,430,440,445,440,435,430],
        "us-west-2": [180,175,170,165,160,158,155,148,130,110,90,80,
                      75,78,85,100,130,155,165,170,175,178,180,182],
        "eu-west-1": [240,235,230,228,225,222,218,215,200,185,175,165,
                      160,162,170,185,200,220,235,245,248,245,242,240],
        "ap-southeast-1": [490,485,480,475,470,465,462,460,455,445,430,
                           420,415,418,425,435,450,465,475,482,488,490,492,491],
    }
    return profiles.get(region, profiles["us-east-1"])

def optimize_batch_schedule(
    jobs: list[dict],
    region: str = "us-east-1",
    max_parallel_jobs: int = 3,
    deadline_hours: int = 24
) -> dict:
    """
    Solve the batch job scheduling optimization problem using Linear Programming.

    jobs: list of dicts with keys: name, duration_hours, cpu_kwh, memory_kwh
    Returns: optimal schedule with assigned start hours
    """
    carbon_intensity = get_hourly_carbon_intensity(region)
    n_jobs = len(jobs)
    hours = list(range(deadline_hours))

    prob = LpProblem("GreenBatchScheduler", LpMinimize)

    # Decision variable: x[j][h] = 1 if job j starts at hour h
    x = [[LpVariable(f"x_{j}_{h}", cat=LpBinary) for h in hours] for j in range(n_jobs)]

    # Objective: minimize total carbon emissions
    total_carbon = []
    for j, job in enumerate(jobs):
        power_kwh = job["cpu_kwh"] + job["memory_kwh"]
        for h in hours:
            # Carbon during job execution (spans duration_hours)
            job_carbon = sum(
                carbon_intensity[(h + t) % 24] * power_kwh
                for t in range(job["duration_hours"])
            )
            total_carbon.append(x[j][h] * job_carbon)

    prob += lpSum(total_carbon)

    # Constraint 1: Each job must be scheduled exactly once
    for j in range(n_jobs):
        prob += lpSum(x[j][h] for h in hours) == 1

    # Constraint 2: Job must finish before deadline
    for j, job in enumerate(jobs):
        for h in hours:
            if h + job["duration_hours"] > deadline_hours:
                prob += x[j][h] == 0

    # Constraint 3: Max parallel jobs per hour
    for h in hours:
        running = []
        for j, job in enumerate(jobs):
            for start_h in range(max(0, h - job["duration_hours"] + 1), h + 1):
                if start_h in hours:
                    running.append(x[j][start_h])
        prob += lpSum(running) <= max_parallel_jobs

    prob.solve()

    results = {"status": LpStatus[prob.status], "schedule": [], "total_carbon_gco2": 0}

    for j, job in enumerate(jobs):
        for h in hours:
            if value(x[j][h]) and value(x[j][h]) > 0.5:
                power_kwh = job["cpu_kwh"] + job["memory_kwh"]
                job_carbon = sum(
                    carbon_intensity[(h + t) % 24] * power_kwh
                    for t in range(job["duration_hours"])
                )
                avg_ci = np.mean([carbon_intensity[(h + t) % 24] for t in range(job["duration_hours"])])
                results["schedule"].append({
                    "job": job["name"],
                    "start_hour": h,
                    "end_hour": h + job["duration_hours"],
                    "duration_hours": job["duration_hours"],
                    "avg_carbon_intensity": round(avg_ci, 1),
                    "carbon_gco2": round(job_carbon, 2),
                    "power_kwh": round(power_kwh, 4)
                })
                results["total_carbon_gco2"] += job_carbon

    results["total_carbon_gco2"] = round(results["total_carbon_gco2"], 2)

    # Compare to naive scheduling (start all at hour 0)
    naive_carbon = 0
    for j, job in enumerate(jobs):
        power_kwh = job["cpu_kwh"] + job["memory_kwh"]
        naive_carbon += sum(carbon_intensity[t % 24] * power_kwh for t in range(job["duration_hours"]))
    results["naive_carbon_gco2"] = round(naive_carbon, 2)
    results["carbon_saved_gco2"] = round(naive_carbon - results["total_carbon_gco2"], 2)
    results["savings_pct"] = round((results["carbon_saved_gco2"] / naive_carbon) * 100, 1)
    results["carbon_intensity_profile"] = carbon_intensity

    return results


if __name__ == "__main__":
    sample_jobs = [
        {"name": "ML Training", "duration_hours": 3, "cpu_kwh": 0.5, "memory_kwh": 0.2},
        {"name": "ETL Pipeline", "duration_hours": 2, "cpu_kwh": 0.3, "memory_kwh": 0.1},
        {"name": "Report Generation", "duration_hours": 1, "cpu_kwh": 0.1, "memory_kwh": 0.05},
        {"name": "Data Backup", "duration_hours": 2, "cpu_kwh": 0.2, "memory_kwh": 0.08},
    ]
    result = optimize_batch_schedule(sample_jobs, region="us-east-1")
    print(f"\nOptimal Schedule (Status: {result['status']})")
    for s in result["schedule"]:
        print(f"  {s['job']}: Start {s['start_hour']}:00 → {s['end_hour']}:00 | {s['carbon_gco2']} gCO2")
    print(f"\nTotal Carbon: {result['total_carbon_gco2']} gCO2")
    print(f"Carbon Saved vs Naive: {result['carbon_saved_gco2']} gCO2 ({result['savings_pct']}%)")