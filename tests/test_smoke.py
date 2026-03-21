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


if __name__ == "__main__":
    unittest.main()
