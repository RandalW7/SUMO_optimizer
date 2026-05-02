import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, LlamaConfig

from optimizers_torch import SUMO
from peft_pretraining.modeling_llama import LlamaForCausalLM
from peft_pretraining.training_utils import get_scheculer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLaMA pretraining on FineWeb with AdamW/SUMO/Muon.")
    parser.add_argument("--model_config", type=str, default="configs/llama_130m.json")
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset_config_name", type=str, default="sample-10BT")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--streaming", action="store_true", default=True)
    parser.add_argument("--no_streaming", action="store_true", help="Disable streaming mode.")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size.")
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--num_training_steps", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, choices=["bf16", "fp32"], default="bf16")

    parser.add_argument("--optimizer", type=str, choices=["adamw", "sumo", "muon"], default="adamw")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--scheduler", type=str, choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    parser.add_argument("--sumo_rank", type=int, default=8)
    parser.add_argument("--sumo_update_proj_gap", type=int, default=200)
    parser.add_argument("--sumo_scale", type=float, default=1.0)
    parser.add_argument("--sumo_proj_type", type=str, default="std")
    parser.add_argument("--sumo_alpha", type=float, default=4.0)
    parser.add_argument("--sumo_gamma", type=float, default=1.1)
    parser.add_argument("--sumo_momentum", type=float, default=0.95)
    parser.add_argument("--sumo_gradient_perpendicular_scale", type=float, default=1.0)

    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_weight_decay", type=float, default=0.1)
    parser.add_argument("--muon_adam_lr", type=float, default=3e-4)
    parser.add_argument("--muon_adam_weight_decay", type=float, default=0.1)
    parser.add_argument("--muon_adam_beta1", type=float, default=0.9)
    parser.add_argument("--muon_adam_beta2", type=float, default=0.95)
    parser.add_argument("--muon_adam_eps", type=float, default=1e-8)

    parser.add_argument("--output_dir", type=str, default="outputs/fineweb_llama130m")
    return parser.parse_args()


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def setup_distributed() -> Dict[str, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return {"local_rank": local_rank, "rank": rank, "world_size": world_size}


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(args: argparse.Namespace, device: torch.device) -> LlamaForCausalLM:
    with open(args.model_config, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    config = LlamaConfig(**cfg_dict)
    model = LlamaForCausalLM(config)
    model.to(device)
    return model


def _non_muon_param(name: str, p: torch.nn.Parameter) -> bool:
    if p.ndim < 2:
        return True
    blocked = ("embed_tokens", "lm_head")
    return any(k in name for k in blocked)


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace):
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        no_decay = ("bias", "norm")
        groups = [
            {
                "params": [p for n, p in named_params if not any(k in n.lower() for k in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in named_params if any(k in n.lower() for k in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(
            groups,
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

    if args.optimizer == "sumo":
        sumo_params = [p for n, p in named_params if p.ndim == 2 and not _non_muon_param(n, p)]
        sumo_ids = {id(p) for p in sumo_params}
        regular_params = [p for _, p in named_params if id(p) not in sumo_ids]
        param_groups = [
            {
                "group_name": "regular_params",
                "params": regular_params,
                "lr": args.learning_rate,
                "beta": (args.beta1, args.beta2),
                "weight_decay_adam": args.weight_decay,
                "lr_adam": args.learning_rate,
                "eps": args.eps,
            },
            {
                "group_name": "sumo_params",
                "params": sumo_params,
                "lr": args.learning_rate,
                "beta": (args.beta1, args.beta2),
                "weight_decay": args.weight_decay,
                "rank": args.sumo_rank,
                "update_proj_gap": args.sumo_update_proj_gap,
                "scale": args.sumo_scale,
                "proj_type": args.sumo_proj_type,
                "alpha": args.sumo_alpha,
                "gamma": args.sumo_gamma,
                "momentum": args.sumo_momentum,
                "gradient_perpendicular_scale": args.sumo_gradient_perpendicular_scale,
                "lr_adam": args.learning_rate,
                "weight_decay_adam": args.weight_decay,
                "eps": args.eps,
            },
        ]
        return SUMO(param_groups)

    muon_repo = Path(__file__).resolve().parents[2] / "Muon"
    if str(muon_repo) not in sys.path:
        sys.path.insert(0, str(muon_repo))
    try:
        from muon import MuonWithAuxAdam
    except ImportError as exc:
        raise RuntimeError("Failed to import Muon. Ensure /home/zwang/Muon is present or install it.") from exc

    muon_params = [p for n, p in named_params if p.ndim >= 2 and not _non_muon_param(n, p)]
    muon_ids = {id(p) for p in muon_params}
    adam_params = [p for _, p in named_params if id(p) not in muon_ids]
    param_groups = [
        {
            "params": muon_params,
            "use_muon": True,
            "lr": args.muon_lr,
            "momentum": args.muon_momentum,
            "weight_decay": args.muon_weight_decay,
        },
        {
            "params": adam_params,
            "use_muon": False,
            "lr": args.muon_adam_lr,
            "betas": (args.muon_adam_beta1, args.muon_adam_beta2),
            "eps": args.muon_adam_eps,
            "weight_decay": args.muon_adam_weight_decay,
        },
    ]
    return MuonWithAuxAdam(param_groups)


def _iter_rank_shard(dataset_iter: Iterable[Dict[str, str]], rank: int, world_size: int) -> Iterator[Dict[str, str]]:
    for idx, ex in enumerate(dataset_iter):
        if idx % world_size == rank:
            yield ex


def token_block_iterator(
    dataset_iter: Iterable[Dict[str, str]],
    tokenizer,
    seq_len: int,
    rank: int,
    world_size: int,
) -> Iterator[List[int]]:
    token_buffer: List[int] = []
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    for example in _iter_rank_shard(dataset_iter, rank=rank, world_size=world_size):
        text = example.get("text", "")
        if not text:
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if eos_id is not None:
            token_ids = token_ids + [eos_id]
        token_buffer.extend(token_ids)
        while len(token_buffer) >= seq_len + 1:
            yield token_buffer[: seq_len + 1]
            token_buffer = token_buffer[seq_len + 1 :]


def batch_iterator(
    dataset_iter: Iterable[Dict[str, str]],
    tokenizer,
    seq_len: int,
    batch_size: int,
    rank: int,
    world_size: int,
) -> Iterator[Batch]:
    curr: List[List[int]] = []
    for block in token_block_iterator(dataset_iter, tokenizer, seq_len, rank, world_size):
        curr.append(block)
        if len(curr) == batch_size:
            x = torch.tensor([b[:-1] for b in curr], dtype=torch.long)
            y = torch.tensor([b[1:] for b in curr], dtype=torch.long)
            attn = torch.ones_like(x, dtype=torch.long)
            yield Batch(input_ids=x, attention_mask=attn, labels=y)
            curr = []


def infinite_batch_iterator(factory) -> Iterator[Batch]:
    while True:
        for item in factory():
            yield item


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    eval_iter: Iterator[Batch],
    eval_steps: int,
    device: torch.device,
) -> float:
    model.eval()
    losses: List[torch.Tensor] = []
    for _ in range(eval_steps):
        batch = next(eval_iter)
        x = batch.input_ids.to(device, non_blocking=True)
        attn = batch.attention_mask.to(device, non_blocking=True)
        y = batch.labels.to(device, non_blocking=True)
        out = model(input_ids=x, attention_mask=attn, labels=y)
        losses.append(out.loss.detach())
    local_loss = torch.stack(losses).mean()
    if dist.is_initialized():
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        local_loss /= dist.get_world_size()
    model.train()
    return local_loss.item()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def main() -> None:
    args = parse_args()
    if args.no_streaming:
        args.streaming = False
    dist_info = setup_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    device = torch.device("cuda", dist_info["local_rank"])
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = create_model(args, device)
    if args.dtype == "bf16":
        model.to(dtype=torch.bfloat16)

    optimizer = build_optimizer(model, args)
    scheduler = get_scheculer(
        optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        cycle_length=args.num_training_steps,
    )

    if world_size > 1:
        ddp_kwargs = {"device_ids": [dist_info["local_rank"]], "output_device": dist_info["local_rank"]}
        if args.optimizer == "muon":
            ddp_kwargs["broadcast_buffers"] = False
        model = DDP(model, **ddp_kwargs)

    train_ds = load_dataset(
        args.dataset_name,
        name=args.dataset_config_name,
        split=args.dataset_split,
        streaming=args.streaming,
    )
    eval_ds = load_dataset(
        args.dataset_name,
        name=args.dataset_config_name,
        split=args.dataset_split,
        streaming=args.streaming,
    )
    if args.streaming:
        # Use a held-out streaming tail as a lightweight validation stream.
        eval_source = eval_ds.skip(10_000)
        train_source = train_ds
    else:
        eval_source = eval_ds.select(range(min(10_000, len(eval_ds))))
        train_source = train_ds

    def train_factory():
        return batch_iterator(
            train_source,
            tokenizer,
            seq_len=args.max_length,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
        )

    def eval_factory():
        return batch_iterator(
            eval_source,
            tokenizer,
            seq_len=args.max_length,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
        )

    train_iter = infinite_batch_iterator(train_factory)
    eval_iter = infinite_batch_iterator(eval_factory)

    running_loss = 0.0
    start_time = time.time()
    best_eval_loss = float("inf")

    for step in range(1, args.num_training_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.gradient_accumulation):
            batch = next(train_iter)
            x = batch.input_ids.to(device, non_blocking=True)
            attn = batch.attention_mask.to(device, non_blocking=True)
            y = batch.labels.to(device, non_blocking=True)
            with (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.dtype == "bf16" else nullcontext()):
                outputs = model(input_ids=x, attention_mask=attn, labels=y)
                loss = outputs.loss / args.gradient_accumulation
            loss.backward()
            accum_loss += loss.item()
        optimizer.step()
        scheduler.step()
        running_loss += accum_loss

        if step % 20 == 0 and is_main_process(rank):
            elapsed = time.time() - start_time
            avg_loss = running_loss / 20
            tok_per_step = args.batch_size * args.max_length * args.gradient_accumulation * world_size
            tok_per_sec = (tok_per_step * 20) / max(elapsed, 1e-6)
            lr = scheduler.get_last_lr()[0]
            print(
                f"[step {step:6d}] train_loss={avg_loss:.4f} lr={lr:.3e} "
                f"tokens/s={tok_per_sec:,.0f}",
                flush=True,
            )
            running_loss = 0.0
            start_time = time.time()

        if step % args.eval_every == 0:
            eval_loss = evaluate(model, eval_iter, args.eval_steps, device)
            if is_main_process(rank):
                ppl = math.exp(min(20.0, eval_loss))
                print(f"[step {step:6d}] eval_loss={eval_loss:.4f} eval_ppl={ppl:.2f}", flush=True)
                best_eval_loss = min(best_eval_loss, eval_loss)

        if step % args.save_every == 0 and is_main_process(rank):
            ckpt_dir = os.path.join(args.output_dir, f"step_{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped = unwrap_model(model)
            unwrapped.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    if is_main_process(rank):
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = unwrap_model(model)
        unwrapped.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        final_metrics = {
            "best_eval_loss": best_eval_loss,
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "steps": args.num_training_steps,
        }
        with open(os.path.join(args.output_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)
        print(f"Saved final metrics to {args.output_dir}/final_metrics.json", flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    main()
