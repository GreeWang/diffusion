import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT_DIR / "experiments" / "ablation_manifest.yaml"
DEFAULT_PROMPTSET = "experiments/promptset_v1.json"
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


MATRIX_PRESETS = {
    "P0": [
        ("baseline", 0, 0, 0, 0),
        ("base+xformers", 0, 1, 0, 0),
        ("base+offload", 0, 0, 1, 0),
        ("base+tiling", 0, 0, 0, 1),
        ("hidiffusion", 1, 0, 0, 0),
        ("hi+xformers", 1, 1, 0, 0),
        ("hi+xformers+tiling", 1, 1, 0, 1),
        ("hi+xformers+offload", 1, 1, 1, 0),
    ],
    "P1": [
        ("hi+tiling", 1, 0, 0, 1),
        ("hi+offload", 1, 0, 1, 0),
    ],
    "P2": [
        ("base+xformers+tiling", 0, 1, 0, 1),
        ("base+xformers+offload", 0, 1, 1, 0),
        ("base+tiling+offload", 0, 0, 1, 1),
        ("hi+tiling+offload", 1, 0, 1, 1),
        ("hi+all", 1, 1, 1, 1),
    ],
}


PROFILES = {
    "smoke": {
        "resolutions": [{"h": 768, "w": 768}, {"h": 1024, "w": 1024}],
        "seeds": [0],
        "repeats": 1,
        "steps": 8,
        "limit_prompts": 1,
        "priorities": ["P0"],
    },
    "p0": {
        "resolutions": [{"h": 1024, "w": 1024}, {"h": 1536, "w": 1536}, {"h": 2048, "w": 2048}],
        "seeds": [0, 1],
        "repeats": 3,
        "steps": 20,
        "limit_prompts": 1,
        "priorities": ["P0"],
    },
    "full": {
        "resolutions": [{"h": 1024, "w": 1024}, {"h": 1536, "w": 1536}, {"h": 2048, "w": 2048}],
        "seeds": [0, 1],
        "repeats": 3,
        "steps": 20,
        "limit_prompts": 2,
        "priorities": ["P0", "P1", "P2"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an ablation experiment manifest.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="p0")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--scheduler", type=str, default="ddim")
    parser.add_argument("--promptset_path", type=str, default=DEFAULT_PROMPTSET)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--limit_prompts", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument("--seed_list", type=str, default=None, help="Comma separated seed override.")
    parser.add_argument("--resolutions", type=str, default=None, help="Comma separated HxW pairs, e.g. 1024x1024,1536x1536.")
    parser.add_argument("--include_priorities", type=str, default=None, help="Comma separated subset such as P0,P1.")
    return parser.parse_args()


def parse_seed_list(raw: str | None, fallback: list[int]) -> list[int]:
    if not raw:
        return fallback
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("seed_list override must not be empty")
    return values


def parse_resolutions(raw: str | None, fallback: list[dict]) -> list[dict]:
    if not raw:
        return fallback
    resolutions = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid resolution override: {item}")
        h_raw, w_raw = item.split("x", 1)
        resolutions.append({"h": int(h_raw), "w": int(w_raw)})
    if not resolutions:
        raise ValueError("resolutions override must not be empty")
    return resolutions


def selected_priorities(raw: str | None, fallback: list[str]) -> list[str]:
    if not raw:
        return fallback
    priorities = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not priorities:
        raise ValueError("include_priorities override must not be empty")
    return priorities


def build_manifest(args: argparse.Namespace) -> dict:
    profile = deepcopy(PROFILES[args.profile])
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    priorities = selected_priorities(args.include_priorities, profile["priorities"])
    manifest = {
        "run_id": run_id,
        "profile": args.profile,
        "model_id": args.model_id,
        "scheduler": args.scheduler,
        "promptset_path": args.promptset_path,
        "resolutions": parse_resolutions(args.resolutions, profile["resolutions"]),
        "steps": args.steps if args.steps is not None else profile["steps"],
        "seeds": parse_seed_list(args.seed_list, profile["seeds"]),
        "repeats": args.repeats if args.repeats is not None else profile["repeats"],
        "limit_prompts": args.limit_prompts if args.limit_prompts is not None else profile["limit_prompts"],
        "guidance_scale": args.guidance_scale,
        "dtype": args.dtype,
        "device": args.device,
        "num_warmup": args.num_warmup,
        "matrix": [],
    }
    for priority in priorities:
        for exp_id, hidiffusion, xformers, cpu_offload, vae_tiling in MATRIX_PRESETS[priority]:
            manifest["matrix"].append(
                {
                    "exp_id": exp_id,
                    "priority": priority,
                    "hidiffusion": hidiffusion,
                    "xformers": xformers,
                    "cpu_offload": cpu_offload,
                    "vae_tiling": vae_tiling,
                }
            )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
