import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class SmokeTests(unittest.TestCase):
    def test_auto_config_writes_strategy_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "strategy.json"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/benchmark/auto_config.py",
                    "--vram_budget_gb",
                    "12",
                    "--height",
                    "2048",
                    "--width",
                    "2048",
                    "--out",
                    str(out)
                ],
                cwd=REPO_ROOT,
                check=True
            )
            payload = json.loads(out.read_text(encoding="utf-8"))
            for key in [
                "enable_hidiffusion",
                "enable_xformers",
                "enable_cpu_offload",
                "enable_vae_tiling",
                "steps"
            ]:
                self.assertIn(key, payload)

    def test_benchmark_dry_run_generates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "results"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/benchmark/run_benchmark.py",
                    "--dry_run",
                    "1",
                    "--height",
                    "512,1024",
                    "--width",
                    "512,1024",
                    "--seeds",
                    "0,1",
                    "--enable_hidiffusion",
                    "1",
                    "--limit_prompts",
                    "2",
                    "--out_dir",
                    str(out_dir)
                ],
                cwd=REPO_ROOT,
                check=True
            )
            self.assertTrue((out_dir / "raw.csv").exists())
            self.assertTrue((out_dir / "report.md").exists())
            self.assertTrue((out_dir / "metadata.json").exists())
            self.assertTrue(any((out_dir / "samples").glob("*.png")))


    def test_ablation_dry_run_pipeline_generates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "ablation_manifest.yaml"
            run_root = Path(tmpdir) / "runs"
            ablation_root = Path(tmpdir) / "ablation"

            subprocess.run(
                [
                    sys.executable,
                    "scripts/ablation/gen_manifest.py",
                    "--profile",
                    "smoke",
                    "--model_id",
                    "runwayml/stable-diffusion-v1-5",
                    "--out",
                    str(manifest_path),
                    "--run_id",
                    "test_run",
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/ablation/run_manifest.py",
                    "--manifest",
                    str(manifest_path),
                    "--output_root",
                    str(run_root),
                    "--dry_run",
                    "1",
                    "--max_jobs",
                    "4",
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/ablation/aggregate.py",
                    "--run_dir",
                    str(run_root / "test_run"),
                    "--out_csv",
                    str(ablation_root / "ablation.csv"),
                    "--out_report",
                    str(ablation_root / "ablation_report.md"),
                    "--recommended_out",
                    str(ablation_root / "recommended_strategy.json"),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    "scripts/ablation/plot.py",
                    "--csv",
                    str(ablation_root / "ablation.csv"),
                    "--out_dir",
                    str(ablation_root / "figures"),
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            self.assertTrue(manifest_path.exists())
            self.assertTrue((run_root / "test_run" / "raw.csv").exists())
            self.assertTrue((ablation_root / "ablation.csv").exists())
            self.assertTrue((ablation_root / "ablation_report.md").exists())
            self.assertTrue((ablation_root / "recommended_strategy.json").exists())
            self.assertTrue(any((ablation_root / "figures").glob("*.png")))


if __name__ == "__main__":
    unittest.main()
