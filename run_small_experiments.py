from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL_NAMES = [
    "mamba_spectrogram",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run small ESC-50 comparison experiments with shared evaluation setup."
    )
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--output-dir", default="runs/small", type=str)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--val-fold", default=1, type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--max-train-steps", default=120, type=int)
    parser.add_argument("--max-val-steps", default=40, type=int)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help=f"Models to run. Choices: {MODEL_NAMES}",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    from .train import TrainConfig, run_training
    from .plotting import save_small_experiment_plots

    default_models = ["mamba_spectrogram"]
    if args.models:
        invalid = [name for name in args.models if name not in MODEL_NAMES]
        if invalid:
            raise ValueError(f"Unsupported model names: {invalid}. Choices: {MODEL_NAMES}")
        models = list(dict.fromkeys(args.models))
    else:
        models = default_models

    summaries: list[dict[str, Any]] = []
    for model_name in models:
        print(f"\n=== Running {model_name} ===")
        cfg = TrainConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            model=model_name,
            val_fold=args.val_fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            device=args.device,
            max_train_steps=args.max_train_steps,
            max_val_steps=args.max_val_steps,
        )
        summary = run_training(cfg)
        summaries.append(summary)

    plot_outputs: dict[str, str] = {}
    try:
        plot_outputs = save_small_experiment_plots(summaries, args.output_dir)
    except ImportError:
        print(
            "Warning: matplotlib is not installed; skipped small experiment comparison plots."
        )

    summary_path = Path(args.output_dir) / "small_experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nSaved summary to {summary_path}")
    if plot_outputs:
        print(f"Saved plots: {json.dumps(plot_outputs, indent=2)}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
