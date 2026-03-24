import argparse
import json
import math
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT_DIR / "results" / "ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ablation raw results into CSV and Markdown report.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_csv", default=str(DEFAULT_RESULTS_ROOT / "ablation.csv"))
    parser.add_argument("--out_report", default=str(DEFAULT_RESULTS_ROOT / "ablation_report.md"))
    parser.add_argument("--recommended_out", default=str(DEFAULT_RESULTS_ROOT / "recommended_strategy.json"))
    return parser.parse_args()


def t_critical_95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        24: 2.064,
        30: 2.042,
    }
    if df <= 0:
        return math.nan
    if df in table:
        return table[df]
    if df < 30:
        nearest = max(k for k in table if k <= df)
        return table[nearest]
    return 1.96


def ci95(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return math.nan
    std = values.std(ddof=1)
    return t_critical_95(len(values) - 1) * (std / math.sqrt(len(values)))


def pct_change(baseline: float, candidate: float) -> float:
    if pd.isna(baseline) or baseline == 0 or pd.isna(candidate):
        return math.nan
    return (baseline - candidate) / baseline * 100.0


def build_recommendation(success_df: pd.DataFrame) -> dict:
    grouped = (
        success_df.groupby(["exp_id", "height", "width"], as_index=False)
        .agg(
            mean_latency=("latency_sec", "mean"),
            mean_peak_vram=("peak_vram_mb", "mean"),
        )
    )
    if grouped.empty:
        return {"status": "no_successful_runs"}
    ranked = grouped.sort_values(["mean_latency", "mean_peak_vram"], ascending=[True, True]).iloc[0]
    return {
        "exp_id": ranked["exp_id"],
        "height": int(ranked["height"]),
        "width": int(ranked["width"]),
        "mean_latency_sec": round(float(ranked["mean_latency"]), 6),
        "mean_peak_vram_mb": round(float(ranked["mean_peak_vram"]), 3),
        "selection_rule": "Lowest average latency among successful runs, breaking ties with lower peak VRAM.",
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    raw_csv = run_dir / "raw.csv"
    df = pd.read_csv(raw_csv)
    df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    success_df = df[df["success"] == 1].copy()
    agg_success = (
        success_df.groupby(["priority", "exp_id", "height", "width"], as_index=False)
        .agg(
            mean_latency=("latency_sec", "mean"),
            ci_latency=("latency_sec", ci95),
            mean_throughput=("throughput_img_s", "mean"),
            ci_throughput=("throughput_img_s", ci95),
            mean_peak_vram=("peak_vram_mb", "mean"),
            ci_peak_vram=("peak_vram_mb", ci95),
            n_success=("success", "count"),
        )
    )
    success_rate = (
        df.groupby(["priority", "exp_id", "height", "width"], as_index=False)
        .agg(total_runs=("success", "count"), success_rate=("success", "mean"))
    )
    summary = agg_success.merge(success_rate, on=["priority", "exp_id", "height", "width"], how="outer")

    failure_counts = (
        df[df["success"] == 0]
        .groupby(["failure_reason"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )

    conclusions = []
    for (height, width), group in summary.groupby(["height", "width"]):
        baseline = group[group["exp_id"] == "baseline"]
        hidiffusion = group[group["exp_id"] == "hidiffusion"]
        if not baseline.empty and not hidiffusion.empty:
            baseline_row = baseline.iloc[0]
            hidiffusion_row = hidiffusion.iloc[0]
            latency_gain = pct_change(baseline_row["mean_latency"], hidiffusion_row["mean_latency"])
            vram_delta = hidiffusion_row["mean_peak_vram"] - baseline_row["mean_peak_vram"]
            if not pd.isna(latency_gain):
                conclusions.append(
                    f"- {height}x{width}: HiDiffusion relative to baseline improved average latency by {latency_gain:.1f}% with peak VRAM delta {vram_delta:.1f} MB."
                )
        hi_xf = group[group["exp_id"] == "hi+xformers"]
        hid = group[group["exp_id"] == "hidiffusion"]
        if not hi_xf.empty and not hid.empty:
            delta = pct_change(hid.iloc[0]["mean_latency"], hi_xf.iloc[0]["mean_latency"])
            if not pd.isna(delta):
                conclusions.append(f"- {height}x{width}: adding xformers on top of HiDiffusion changed latency by {delta:.1f}%.")

    recommendation = build_recommendation(success_df)
    recommended_path = Path(args.recommended_out)
    recommended_path.parent.mkdir(parents=True, exist_ok=True)
    recommended_path.write_text(json.dumps(recommendation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total_rows = len(df)
    successful_rows = int(df["success"].sum())
    lines = [
        "# Ablation Report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Total rows: {total_rows}",
        f"- Successful rows: {successful_rows}",
        "",
        "## Aggregate Metrics",
        "",
        "| priority | exp_id | resolution | mean latency (s) | 95% CI | mean throughput (img/s) | mean peak VRAM (MB) | success rate | successful runs | total runs |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary.sort_values(["priority", "height", "width", "exp_id"]).to_dict("records"):
        resolution = f"{int(row['height'])}x{int(row['width'])}"
        n_success = 0 if pd.isna(row.get('n_success')) else int(row.get('n_success'))
        total_runs = 0 if pd.isna(row.get('total_runs')) else int(row.get('total_runs'))
        lines.append(
            f"| {row['priority']} | {row['exp_id']} | {resolution} | "
            f"{row.get('mean_latency', math.nan):.3f} | {row.get('ci_latency', math.nan):.3f} | "
            f"{row.get('mean_throughput', math.nan):.3f} | {row.get('mean_peak_vram', math.nan):.1f} | "
            f"{row.get('success_rate', math.nan):.2%} | {n_success} | {total_runs} |"
        )

    lines.extend(["", "## Main Findings", ""])
    if conclusions:
        lines.extend(conclusions)
    else:
        lines.append("- No paired baseline/HiDiffusion conclusions could be inferred from the available successful runs.")

    lines.extend(["", "## Failure Breakdown", ""])
    if failure_counts.empty:
        lines.append("- No failures recorded.")
    else:
        for row in failure_counts.to_dict("records"):
            lines.append(f"- `{row['failure_reason']}`: {int(row['size'])}")

    lines.extend(["", "## Recommendation", ""])
    if recommendation.get("status") == "no_successful_runs":
        lines.append("- No successful runs were available, so no recommended strategy could be derived.")
    else:
        exp_id = recommendation["exp_id"]
        height = recommendation["height"]
        width = recommendation["width"]
        latency = recommendation["mean_latency_sec"]
        vram = recommendation["mean_peak_vram_mb"]
        lines.append(
            f"- Recommended configuration: `{exp_id}` at {height}x{width} with mean latency {latency:.3f}s and peak VRAM {vram:.1f} MB."
        )

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_report}")
    print(f"Wrote {recommended_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
