import argparse
import itertools
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_SEARCH_SPACE = {
    "adamw": {
        "learning_rate": [2e-4, 3e-4, 4e-4],
        "weight_decay": [0.05, 0.1],
        "warmup_steps": [100, 200],
    },
    "sumo": {
        "learning_rate": [1.5e-4, 2e-4, 3e-4],
        "weight_decay": [0.05, 0.1],
        "sumo_rank": [8, 16],
        "sumo_update_proj_gap": [100, 200],
    },
    "muon": {
        "muon_lr": [0.01, 0.02, 0.03],
        "muon_weight_decay": [0.05, 0.1],
        "muon_adam_lr": [2e-4, 3e-4],
        "muon_adam_weight_decay": [0.05, 0.1],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search launcher for 130M LLaMA FineWeb pretraining.")
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--model_config", type=str, default="configs/llama_130m.json")
    parser.add_argument("--output_root", type=str, default="outputs/fineweb_130m_grid")
    parser.add_argument("--nproc_per_node", type=int, default=4)
    parser.add_argument("--search_space_file", type=str, default=None)
    parser.add_argument("--max_trials", type=int, default=0, help="0 means run all generated trials.")
    parser.add_argument("--dry_run", action="store_true")

    # Shared training arguments for each trial.
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset_config_name", type=str, default="sample-10BT")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--num_training_steps", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_search_space(path: str | None) -> Dict[str, Dict[str, List[float]]]:
    if path is None:
        return DEFAULT_SEARCH_SPACE
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expand_dict_grid(grid_dict: Dict[str, List]) -> Iterable[Dict]:
    keys = sorted(grid_dict.keys())
    values = [grid_dict[k] if isinstance(grid_dict[k], list) else [grid_dict[k]] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def build_common_args(args: argparse.Namespace) -> List[str]:
    return [
        "--model_config",
        args.model_config,
        "--tokenizer_name_or_path",
        args.tokenizer_name_or_path,
        "--dataset_name",
        args.dataset_name,
        "--dataset_config_name",
        args.dataset_config_name,
        "--dataset_split",
        args.dataset_split,
        "--max_length",
        str(args.max_length),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation",
        str(args.gradient_accumulation),
        "--num_training_steps",
        str(args.num_training_steps),
        "--eval_every",
        str(args.eval_every),
        "--eval_steps",
        str(args.eval_steps),
        "--save_every",
        str(args.save_every),
        "--dtype",
        args.dtype,
        "--scheduler",
        args.scheduler,
        "--min_lr_ratio",
        str(args.min_lr_ratio),
        "--seed",
        str(args.seed),
    ]


def trial_to_cli_args(trial: Dict) -> List[str]:
    cli_args: List[str] = []
    for k, v in sorted(trial.items()):
        cli_args.extend([f"--{k}", str(v)])
    return cli_args


def main() -> None:
    args = parse_args()
    search_space = load_search_space(args.search_space_file)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_trials: List[Dict] = []
    for optimizer_name in ("adamw", "sumo", "muon"):
        opt_space = search_space.get(optimizer_name, {})
        for params in expand_dict_grid(opt_space):
            trial = {"optimizer": optimizer_name, **params}
            all_trials.append(trial)

    if args.max_trials > 0:
        all_trials = all_trials[: args.max_trials]

    print(f"Generated {len(all_trials)} trials.")
    common_args = build_common_args(args)
    results = []

    for i, trial in enumerate(all_trials, start=1):
        trial_name = f"trial_{i:03d}_{trial['optimizer']}"
        trial_dir = output_root / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(args.nproc_per_node),
            "peft_pretraining/train_fineweb_llama.py",
            *common_args,
            "--output_dir",
            str(trial_dir),
            *trial_to_cli_args(trial),
        ]
        pretty_cmd = " ".join(shlex.quote(x) for x in cmd)
        print(f"\n[{i}/{len(all_trials)}] {trial_name}")
        print(pretty_cmd)

        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)
        metrics_path = trial_dir / "final_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            results.append({"trial": trial_name, "params": trial, "metrics": metrics})
        else:
            results.append({"trial": trial_name, "params": trial, "metrics": {"best_eval_loss": float("inf")}})

    if args.dry_run:
        return

    results = sorted(results, key=lambda x: x["metrics"].get("best_eval_loss", float("inf")))
    results_path = output_root / "grid_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if results:
        best = results[0]
        print("\nBest trial:")
        print(json.dumps(best, indent=2))
        print(f"\nSaved full results to: {results_path}")


if __name__ == "__main__":
    main()
