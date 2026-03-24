import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FIG_DIR = ROOT_DIR / "results" / "ablation" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation figures from ablation.csv.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", default=str(DEFAULT_FIG_DIR))
    return parser.parse_args()


def save_placeholder(path: Path, title: str, lines: list[str]) -> None:
    image = Image.new("RGB", (1200, 720), color=(248, 249, 251))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 1170, 690), outline=(70, 92, 120), width=4)
    draw.text((60, 60), title, fill=(35, 45, 60))
    y = 120
    for line in lines:
        draw.text((60, y), line, fill=(55, 65, 82))
        y += 34
    image.save(path)


def save_grouped_bar(df: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    resolutions = sorted(df["resolution"].unique())
    fig, axes = plt.subplots(len(resolutions), 1, figsize=(14, max(4, 3.2 * len(resolutions))), sharex=True)
    if len(resolutions) == 1:
        axes = [axes]
    for ax, resolution in zip(axes, resolutions):
        subset = df[df["resolution"] == resolution].sort_values("exp_id")
        ax.bar(subset["exp_id"], subset[metric], color="#4E79A7")
        ax.set_title(resolution)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
    axes[-1].set_xlabel("experiment")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_success_rate(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_id, subset in df.groupby("exp_id"):
        ax.plot(subset["resolution"], subset["success_rate"], marker="o", label=exp_id)
    ax.set_ylabel("success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("resolution")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_pareto(df: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_id, subset in df.groupby("exp_id"):
        ax.scatter(subset["mean_peak_vram"], subset["mean_latency"], label=exp_id)
        for _, row in subset.iterrows():
            ax.annotate(row["resolution"], (row["mean_peak_vram"], row["mean_latency"]), fontsize=7)
    ax.set_xlabel("mean peak VRAM (MB)")
    ax.set_ylabel("mean latency (s)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_placeholder_suite(out_dir: Path, agg: pd.DataFrame, success_rate: pd.DataFrame) -> None:
    top_rows = agg.sort_values(["resolution", "exp_id"]).head(10)
    summary_lines = [
        f"rows={len(agg)} unique_experiments={agg['exp_id'].nunique() if not agg.empty else 0}",
        f"success_rate_rows={len(success_rate)}",
    ]
    for _, row in top_rows.iterrows():
        summary_lines.append(
            f"{row['exp_id']} {row['resolution']} latency={row['mean_latency']:.3f}s vram={row['mean_peak_vram']:.1f}MB"
        )
    save_placeholder(out_dir / "latency_by_resolution.png", "Latency Placeholder", summary_lines)
    save_placeholder(out_dir / "peak_vram_by_resolution.png", "Peak VRAM Placeholder", summary_lines)
    rate_lines = [
        f"{row.exp_id} {row.resolution} success_rate={row.success_rate:.2%}"
        for row in success_rate.sort_values(["resolution", "exp_id"]).head(12).itertuples()
    ]
    save_placeholder(out_dir / "success_rate_by_resolution.png", "Success Rate Placeholder", rate_lines)
    save_placeholder(out_dir / "pareto_latency_vs_vram.png", "Pareto Placeholder", summary_lines)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.csv)
    df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)
    df["resolution"] = df["height"].astype(int).astype(str) + "x" + df["width"].astype(int).astype(str)

    success_df = df[df["success"] == 1].copy()
    agg = (
        success_df.groupby(["exp_id", "height", "width", "resolution"], as_index=False)
        .agg(
            mean_latency=("latency_sec", "mean"),
            mean_peak_vram=("peak_vram_mb", "mean"),
        )
    )
    success_rate = (
        df.groupby(["exp_id", "height", "width", "resolution"], as_index=False)
        .agg(success_rate=("success", "mean"))
        .sort_values(["height", "width", "exp_id"])
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib  # noqa: F401

        save_grouped_bar(agg, "mean_latency", "latency (s)", out_dir / "latency_by_resolution.png")
        save_grouped_bar(agg, "mean_peak_vram", "peak VRAM (MB)", out_dir / "peak_vram_by_resolution.png")
        save_success_rate(success_rate, out_dir / "success_rate_by_resolution.png")
        save_pareto(agg, out_dir / "pareto_latency_vs_vram.png")
    except ModuleNotFoundError:
        save_placeholder_suite(out_dir, agg, success_rate)

    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
