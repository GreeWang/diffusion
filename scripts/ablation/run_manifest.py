import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
RUN_ONE = ROOT_DIR / "scripts" / "ablation" / "run_one.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand and run the ablation manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", default="results/ablation/runs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", type=int, choices=[0, 1], default=0)
    parser.add_argument("--mock_pipe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max_jobs", type=int, default=None, help="Optional limit for smoke/debug runs.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_jobs(manifest: dict) -> list[dict]:
    jobs = []
    limit_prompts = int(manifest.get("limit_prompts", 1))
    for experiment in manifest["matrix"]:
        for resolution in manifest["resolutions"]:
            for prompt_id in range(limit_prompts):
                for seed in manifest["seeds"]:
                    for repeat_idx in range(manifest["repeats"]):
                        jobs.append(
                            {
                                "exp_id": experiment["exp_id"],
                                "priority": experiment["priority"],
                                "hidiffusion": int(experiment["hidiffusion"]),
                                "xformers": int(experiment["xformers"]),
                                "cpu_offload": int(experiment["cpu_offload"]),
                                "vae_tiling": int(experiment["vae_tiling"]),
                                "height": int(resolution["h"]),
                                "width": int(resolution["w"]),
                                "prompt_id": prompt_id,
                                "seed": int(seed),
                                "repeat_idx": int(repeat_idx),
                            }
                        )
    return jobs


def write_jobs_csv(path: Path, jobs: list[dict]) -> None:
    if not jobs:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(jobs[0].keys()))
        writer.writeheader()
        writer.writerows(jobs)


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    run_id = manifest["run_id"]
    run_dir = Path(args.output_root) / run_id
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(manifest)
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    (run_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_jobs_csv(run_dir / "jobs.csv", jobs)

    summary = {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "run_dir": str(run_dir),
        "total_jobs": len(jobs),
        "completed_jobs": 0,
        "failed_subprocesses": 0,
    }

    for idx, job in enumerate(jobs):
        exp_id = job["exp_id"]
        height = job["height"]
        width = job["width"]
        prompt_id = job["prompt_id"]
        seed = job["seed"]
        repeat_idx = job["repeat_idx"]
        job_id = f"{idx:04d}_{exp_id}_{height}x{width}_p{prompt_id}_s{seed}_r{repeat_idx}"
        cmd = [
            args.python,
            str(RUN_ONE),
            "--run_id",
            str(run_id),
            "--exp_id",
            exp_id,
            "--priority",
            job["priority"],
            "--job_id",
            job_id,
            "--model_id",
            manifest["model_id"],
            "--scheduler",
            manifest["scheduler"],
            "--prompt_file",
            manifest["promptset_path"],
            "--prompt_id",
            str(prompt_id),
            "--seed",
            str(seed),
            "--repeat_idx",
            str(repeat_idx),
            "--hidiffusion",
            str(job["hidiffusion"]),
            "--xformers",
            str(job["xformers"]),
            "--cpu_offload",
            str(job["cpu_offload"]),
            "--vae_tiling",
            str(job["vae_tiling"]),
            "--height",
            str(height),
            "--width",
            str(width),
            "--steps",
            str(manifest["steps"]),
            "--guidance_scale",
            str(manifest.get("guidance_scale", 7.5)),
            "--device",
            manifest.get("device", "cuda"),
            "--dtype",
            manifest.get("dtype", "fp16"),
            "--dry_run",
            str(args.dry_run),
            "--mock_pipe",
            str(args.mock_pipe),
            "--out_dir",
            str(run_dir),
        ]
        result = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, check=False)
        log_path = logs_dir / f"{job_id}.log"
        log_path.write_text(result.stdout + ("\n" + result.stderr if result.stderr else ""), encoding="utf-8")
        summary["completed_jobs"] += 1
        if result.returncode != 0:
            summary["failed_subprocesses"] += 1

    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {run_dir / 'raw.csv'}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
