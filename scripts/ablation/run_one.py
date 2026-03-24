import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.benchmark.run_benchmark import (
    collect_environment_metadata,
    dry_run_metrics,
    load_prompts,
    make_mock_image,
    measure_generate,
    sanitize,
    save_image,
)


FIELDS = [
    "run_id",
    "exp_id",
    "priority",
    "job_id",
    "model_id",
    "scheduler",
    "prompt_id",
    "prompt_name",
    "seed",
    "repeat_idx",
    "hidiffusion",
    "xformers",
    "cpu_offload",
    "vae_tiling",
    "height",
    "width",
    "steps",
    "guidance_scale",
    "device",
    "dtype",
    "latency_sec",
    "throughput_img_s",
    "peak_vram_mb",
    "success",
    "failure_reason",
    "torch_version",
    "diffusers_version",
    "hidiffusion_version",
    "xformers_version",
    "python_version",
    "platform",
    "gpu_name",
    "timestamp",
    "sample_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ablation configuration in an isolated process.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--exp_id", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--scheduler", default="ddim")
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--prompt_id", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat_idx", type=int, required=True)
    parser.add_argument("--hidiffusion", type=int, choices=[0, 1], default=0)
    parser.add_argument("--xformers", type=int, choices=[0, 1], default=0)
    parser.add_argument("--cpu_offload", type=int, choices=[0, 1], default=0)
    parser.add_argument("--vae_tiling", type=int, choices=[0, 1], default=0)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--dry_run", type=int, choices=[0, 1], default=0)
    parser.add_argument("--mock_pipe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--sample_only_on_success", type=int, choices=[0, 1], default=1)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def detect_xformers_version() -> str:
    try:
        import xformers

        return getattr(xformers, "__version__", "unknown")
    except Exception:
        return "unavailable"


def build_pipeline_with_failures(args: argparse.Namespace):
    import torch
    from diffusers import AutoPipelineForText2Image, DDIMScheduler

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    pipe_kwargs = {"torch_dtype": torch_dtype}
    if args.dtype == "fp16":
        pipe_kwargs["variant"] = "fp16"
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(args.model_id, **pipe_kwargs)
    except Exception:
        pipe_kwargs.pop("variant", None)
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(args.model_id, **pipe_kwargs)
        except Exception as exc:
            return None, "model_load_failure", exc

    if args.scheduler.lower() == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        return None, "invalid_config", ValueError(f"Unsupported scheduler: {args.scheduler}")

    if args.device == "cuda":
        pipe = pipe.to("cuda")

    if args.xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            return None, "xformers_unavailable", exc
    if args.cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
        except Exception as exc:
            return None, "cpu_offload_hook_error", exc
    if args.vae_tiling:
        try:
            pipe.enable_vae_tiling()
        except Exception as exc:
            return None, "vae_tiling_error", exc
    if args.hidiffusion:
        try:
            from hidiffusion import apply_hidiffusion

            apply_hidiffusion(pipe)
        except Exception as exc:
            return None, "hidiffusion_apply_error", exc
    return pipe, "", None


def classify_failure(reason: str, exc: Exception | None) -> str:
    if reason:
        return reason
    if exc is None:
        return ""
    return f"other_exception:{type(exc).__name__}"


def make_failure_row(args: argparse.Namespace, prompt_name: str, metadata: dict, sample_path: str, reason: str) -> dict:
    return {
        "run_id": args.run_id,
        "exp_id": args.exp_id,
        "priority": args.priority,
        "job_id": args.job_id,
        "model_id": args.model_id,
        "scheduler": args.scheduler,
        "prompt_id": args.prompt_id,
        "prompt_name": prompt_name,
        "seed": args.seed,
        "repeat_idx": args.repeat_idx,
        "hidiffusion": args.hidiffusion,
        "xformers": args.xformers,
        "cpu_offload": args.cpu_offload,
        "vae_tiling": args.vae_tiling,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "device": args.device,
        "dtype": args.dtype,
        "latency_sec": math.nan,
        "throughput_img_s": math.nan,
        "peak_vram_mb": math.nan,
        "success": 0,
        "failure_reason": reason,
        "torch_version": metadata["torch_version"],
        "diffusers_version": metadata["diffusers_version"],
        "hidiffusion_version": metadata["hidiffusion_version"],
        "xformers_version": detect_xformers_version(),
        "python_version": metadata["python_version"],
        "platform": metadata["platform"],
        "gpu_name": metadata["device_name"],
        "timestamp": metadata["timestamp"],
        "sample_path": sample_path,
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    prompts, negative_prompt = load_prompts(args.prompt_file)
    if args.prompt_id < 0 or args.prompt_id >= len(prompts):
        raise ValueError(f"prompt_id {args.prompt_id} is out of range for {args.prompt_file}")
    prompt_item = prompts[args.prompt_id]
    prompt_name = prompt_item["name"]
    metadata = collect_environment_metadata()

    sample_stem = (
        f"{args.exp_id}_p{args.prompt_id}_h{args.height}_w{args.width}_"
        f"s{args.seed}_r{args.repeat_idx}"
    )
    sample_path = out_dir / "samples" / f"{sanitize(sample_stem)}.png"
    row_csv_path = out_dir / "raw.csv"

    if args.dry_run or args.mock_pipe:
        metrics = dry_run_metrics(
            height=args.height,
            width=args.width,
            steps=args.steps,
            enable_hidiffusion=args.hidiffusion,
            seed=args.seed,
        )
        latency = metrics["latency_sec"]
        peak_vram = metrics["peak_vram_mb"]
        if args.xformers:
            latency = round(latency * 0.92, 6)
            peak_vram = round(peak_vram * 0.93, 3)
        if args.cpu_offload:
            latency = round(latency * 1.08, 6)
            peak_vram = round(peak_vram * 0.72, 3)
        if args.vae_tiling and max(args.height, args.width) > 512:
            latency = round(latency * 1.03, 6)
            peak_vram = round(peak_vram * 0.81, 3)
        throughput = round(1.0 / latency, 6) if latency > 0 else math.nan
        image = make_mock_image(prompt_name, args.height, args.width, args.seed, args.hidiffusion)
        save_image(image, sample_path)
        row = {
            "run_id": args.run_id,
            "exp_id": args.exp_id,
            "priority": args.priority,
            "job_id": args.job_id,
            "model_id": args.model_id,
            "scheduler": args.scheduler,
            "prompt_id": args.prompt_id,
            "prompt_name": prompt_name,
            "seed": args.seed,
            "repeat_idx": args.repeat_idx,
            "hidiffusion": args.hidiffusion,
            "xformers": args.xformers,
            "cpu_offload": args.cpu_offload,
            "vae_tiling": args.vae_tiling,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "device": args.device,
            "dtype": args.dtype,
            "latency_sec": latency,
            "throughput_img_s": throughput,
            "peak_vram_mb": peak_vram,
            "success": 1,
            "failure_reason": "",
            "torch_version": metadata["torch_version"],
            "diffusers_version": metadata["diffusers_version"],
            "hidiffusion_version": metadata["hidiffusion_version"],
            "xformers_version": detect_xformers_version(),
            "python_version": metadata["python_version"],
            "platform": metadata["platform"],
            "gpu_name": metadata["device_name"],
            "timestamp": metadata["timestamp"],
            "sample_path": str(sample_path.relative_to(out_dir)),
        }
    else:
        pipe, reason, exc = build_pipeline_with_failures(args)
        if pipe is None:
            row = make_failure_row(
                args=args,
                prompt_name=prompt_name,
                metadata=metadata,
                sample_path="",
                reason=classify_failure(reason, exc),
            )
        else:
            image, metrics = measure_generate(
                pipe,
                prompt=prompt_item["prompt"],
                negative_prompt=negative_prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
            )
            failure_reason = metrics["fail_reason"]
            if failure_reason.startswith("exception:"):
                failure_reason = f"other_exception:{failure_reason.split(':', 1)[1]}"
            if failure_reason == "oom":
                failure_reason = "oom"
            if image is not None:
                save_image(image, sample_path)
                sample_rel_path = str(sample_path.relative_to(out_dir))
            elif not args.sample_only_on_success:
                placeholder = make_mock_image(prompt_name, args.height, args.width, args.seed, args.hidiffusion)
                save_image(placeholder, sample_path)
                sample_rel_path = str(sample_path.relative_to(out_dir))
            else:
                sample_rel_path = ""
            row = {
                "run_id": args.run_id,
                "exp_id": args.exp_id,
                "priority": args.priority,
                "job_id": args.job_id,
                "model_id": args.model_id,
                "scheduler": args.scheduler,
                "prompt_id": args.prompt_id,
                "prompt_name": prompt_name,
                "seed": args.seed,
                "repeat_idx": args.repeat_idx,
                "hidiffusion": args.hidiffusion,
                "xformers": args.xformers,
                "cpu_offload": args.cpu_offload,
                "vae_tiling": args.vae_tiling,
                "height": args.height,
                "width": args.width,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "device": args.device,
                "dtype": args.dtype,
                "latency_sec": metrics["latency_sec"],
                "throughput_img_s": metrics["throughput_img_s"],
                "peak_vram_mb": metrics["peak_vram_mb"],
                "success": int(bool(metrics["success"])),
                "failure_reason": failure_reason,
                "torch_version": metadata["torch_version"],
                "diffusers_version": metadata["diffusers_version"],
                "hidiffusion_version": metadata["hidiffusion_version"],
                "xformers_version": detect_xformers_version(),
                "python_version": metadata["python_version"],
                "platform": metadata["platform"],
                "gpu_name": metadata["device_name"],
                "timestamp": metadata["timestamp"],
                "sample_path": sample_rel_path,
            }

    append_row(row_csv_path, row)
    print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
