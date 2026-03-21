import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend a HiDiffusion inference strategy.")
    parser.add_argument("--vram_budget_gb", type=float, required=True)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--target_resolution", type=int, default=None, help="Square resolution shortcut, e.g. 2048.")
    parser.add_argument("--prefer_speed", type=int, choices=[0, 1], default=1)
    parser.add_argument("--model_family", choices=["sdxl", "sd15"], default="sdxl")
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def resolve_dimensions(height: int | None, width: int | None, target_resolution: int | None) -> tuple[int, int]:
    if target_resolution is not None:
        return target_resolution, target_resolution
    if height is None or width is None:
        raise ValueError("Provide either --target_resolution or both --height and --width.")
    return height, width


def recommend_strategy(vram_budget_gb: float, h: int, w: int, prefer_speed: bool = True, model_family: str = "sdxl") -> dict:
    px = h * w
    mp = px / 1e6
    is_sdxl = model_family == "sdxl"

    strategy = {
        "enable_hidiffusion": False,
        "enable_xformers": True,
        "enable_cpu_offload": False,
        "enable_vae_tiling": False,
        "steps": 20 if is_sdxl else 25,
    }

    if mp >= 1.5:
        strategy["enable_hidiffusion"] = True
    if mp >= (3.0 if is_sdxl else 2.0):
        strategy["enable_vae_tiling"] = True
    if prefer_speed and mp >= 3.0:
        strategy["steps"] = min(strategy["steps"], 12 if is_sdxl else 16)

    if vram_budget_gb <= 8:
        strategy["enable_cpu_offload"] = True
        strategy["enable_vae_tiling"] = True
        strategy["enable_hidiffusion"] = True
        strategy["steps"] = min(strategy["steps"], 8 if prefer_speed else 10)
    elif vram_budget_gb <= 12:
        strategy["enable_hidiffusion"] = True
        if mp >= 3.0:
            strategy["enable_cpu_offload"] = True
        strategy["steps"] = min(strategy["steps"], 12 if prefer_speed else 16)
    elif vram_budget_gb >= 24:
        strategy["enable_cpu_offload"] = False
        strategy["steps"] = max(strategy["steps"], 20 if is_sdxl else 25)

    return {
        **strategy,
        "height": h,
        "width": w,
        "target_megapixels": round(mp, 3),
        "vram_budget_gb": vram_budget_gb,
        "prefer_speed": prefer_speed,
        "model_family": model_family,
        "notes": [
            "Thresholds are rule-based and intended to be calibrated with your benchmark CSV.",
            "When xformers is unavailable, the runner should fall back gracefully and record that metadata."
        ]
    }


def main() -> None:
    args = parse_args()
    height, width = resolve_dimensions(args.height, args.width, args.target_resolution)
    strategy = recommend_strategy(
        vram_budget_gb=args.vram_budget_gb,
        h=height,
        w=width,
        prefer_speed=bool(args.prefer_speed),
        model_family=args.model_family,
    )
    payload = json.dumps(strategy, indent=2, ensure_ascii=False)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
