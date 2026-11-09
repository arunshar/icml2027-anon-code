#!/usr/bin/env python3
"""Train matched completion-level GRPO and learned-critic PPO on strict capacity.

The historical trainer remains unchanged. This protocol uses full-softmax sampling,
frozen rollout quantities, multiple actor epochs, and held-out final evaluation.
The actor has no reference-policy penalty. Every completion retains its raw text.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np
import torch

import rewards
import revision_task as task
from revision_ppo import (FrozenBanditBatch, PromptValueCritic, freeze_bandit_batch,
                          ppo_actor_loss, snapshot_values, value_loss)
from train_speed_grpo import (build_models, chat_ids, completion_mask, device_name,
                              git_dirty, git_sha, param_checksum, pick_device,
                              rng_state_restore, rng_state_snapshot,
                              sample_completions, seed_everything, seq_logprobs)

PROTOCOL_VERSION = "strict_capacity_training_v1"
EVALUATION_BATCH_SIZE = 4
SOURCE_FILES = ("train_revision.py", "revision_task.py", "revision_ppo.py",
                "train_speed_grpo.py", "rewards.py", "tasks.py", "capacity_task.py",
                "speed_task.py")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=("grpo", "ppo"), required=True)
    parser.add_argument("--reward", choices=("binary", "graded"), default="binary")
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--reward-scale", type=float, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--critic-steps", type=int, default=4)
    parser.add_argument("--critic-lr", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--score-microbatch", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=30)
    parser.add_argument("--eval-samples", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--skip-heldout", action="store_true",
                        help="Timing smokes omit held-out performance. Study runs must keep this false.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--no-ref", action="store_true", default=True,
                        help="This protocol always omits a reference model and uses beta=0.")
    return parser


def validate_args(args, dataset):
    for name in ("group_size", "steps", "epochs", "critic_steps", "score_microbatch",
                 "eval_every", "eval_samples", "max_new_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive.")
    if args.group_size < 2:
        raise ValueError("The group size must be at least two for sample standard deviation.")
    if args.seed < 0 or args.eval_seed < 0:
        raise ValueError("Random seeds must be nonnegative.")
    for name in ("weight", "weight_decay", "reward_scale", "lr", "critic_lr",
                 "clip_coef", "grad_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if min(args.reward_scale, args.lr, args.critic_lr, args.grad_clip) <= 0:
        raise ValueError("Reward scale, learning rates, and gradient clipping must be positive.")
    if not 0 < args.clip_coef < 1:
        raise ValueError("The clipping coefficient must lie strictly between zero and one.")
    required_scale = max(1 + args.weight, *(row["maximum_task_value"]
                         for row in dataset["train"]))
    if args.reward_scale < required_scale:
        raise ValueError("The fixed reward scale must bound the binary returns for this weight and dataset.")
    # Explicit generation settings override the model's saved generation defaults.
    args.temperature, args.top_p, args.top_k, args.repetition_penalty = 1.0, 1.0, 0, 1.0
    args.no_ref = True


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprints():
    base = Path(__file__).resolve().parent
    return {name: _file_digest(base / name) for name in SOURCE_FILES}


def _write_json(path, value):
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = Path(str(path) + ".tmp")
    with temporary.open("x") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append(handle, value):
    handle.write(_encoded(value).decode() + "\n")


def summarize_records(records):
    """Unconditional endpoints retain all outputs in their denominators."""
    if not records:
        raise ValueError("An evaluation requires at least one completion.")
    count = len(records)
    present = [row for row in records if row["format_valid"]]
    nonempty = [row for row in records if row["nonempty"]]
    useful = [row for row in records if row["useful_feasible"]]
    average = lambda key, rows: float(np.mean([row[key] for row in rows])) if rows else None
    return {
        "n_completions": count, "n_format_valid": len(present),
        "n_nonempty": len(nonempty), "n_useful_feasible": len(useful),
        "format_valid_rate": len(present) / count,
        "parse_fail_rate": 1 - len(present) / count,
        "nonempty_rate": len(nonempty) / count,
        "empty_selection_rate": sum(row["format_valid"] and not row["nonempty"]
                                    for row in records) / count,
        "useful_feasible_rate": len(useful) / count,
        "useful_failure_rate": 1 - len(useful) / count,
        "capacity_violation_rate_given_format_valid": (sum(not row["capacity_feasible"]
            for row in present) / len(present)) if present else None,
        "n_capacity_violating": sum(not row["capacity_feasible"] for row in present),
        "completion_tokens_mean": average("n_completion_tokens", records),
        "useful_value_mean": average("useful_value", records),
        "normalized_value_mean_given_format_valid": average("normalized_value", present),
        "useful_value_mean_given_useful_feasible": average("useful_value", useful),
        "capacity_residual_mean_given_format_valid": average("capacity_residual", present),
        "reward_mean": average("reward", records),
        "scaled_reward_mean": average("scaled_reward", records),
    }


def _record(gen, mask, plen, text, instance, args, sample_index, step, phase, split):
    score = task.score_completion(text, instance, arm=args.reward, weight=args.weight)
    n_tokens = int(mask.sum().item())
    return {**score, "step": step, "phase": phase, "split": split,
            "sample_index": sample_index, "scaled_reward": score["reward"] / args.reward_scale,
            "completion_token_ids": gen[plen:plen + n_tokens].detach().cpu().tolist(),
            "n_completion_tokens": n_tokens}


@dataclass
class RolloutGroup:
    sequences: torch.Tensor
    mask: torch.Tensor
    batch: FrozenBanditBatch
    records: list


def _scoring_slices(size, microbatch):
    return (slice(start, min(start + microbatch, size)) for start in range(0, size, microbatch))


def collect_rollout(model, tok, prompts, instances, args, eos_ids, critic, step):
    """Freeze old critic values before sampling, then freeze all behavior probabilities."""
    model.eval()
    device = prompts[0].device
    ids = torch.arange(len(instances), device=device).repeat_interleave(args.group_size)
    old_values = (snapshot_values(critic, ids) if critic is not None
                  else torch.zeros(len(ids), device=device))
    groups = []
    with torch.no_grad():
        for index, (prompt, instance) in enumerate(zip(prompts, instances)):
            plen = prompt.shape[1]
            sequences = sample_completions(model, tok, prompt, args, args.group_size)
            mask = completion_mask(sequences, plen, eos_ids)
            records = [_record(sequence, item_mask, plen,
                               tok.decode(sequence[plen:], skip_special_tokens=True),
                               instance, args, sample_index, step, "train", "train")
                       for sample_index, (sequence, item_mask) in enumerate(zip(sequences, mask))]
            behavior = torch.cat([seq_logprobs(model, sequences[part], mask[part])
                                  for part in _scoring_slices(len(sequences), args.score_microbatch)])
            returns = torch.tensor([record["scaled_reward"] for record in records],
                                   device=device, dtype=behavior.dtype)
            start = index * args.group_size
            batch = freeze_bandit_batch(ids[start:start + args.group_size], behavior, returns,
                                         old_values[start:start + args.group_size].to(behavior.dtype))
            if args.method == "grpo":
                advantages = rewards.advantage(np.asarray([row["scaled_reward"] for row in records]))
                batch = replace(batch, advantages=torch.tensor(advantages, dtype=behavior.dtype,
                                                               device=device))
            for sample_index, record in enumerate(records):
                record.update({"behavior_log_prob": float(batch.behavior_log_probs[sample_index]),
                               "old_value": float(batch.old_values[sample_index]),
                               "advantage": float(batch.advantages[sample_index])})
            groups.append(RolloutGroup(sequences, mask, batch, records))
    return groups


def actor_update(model, optimizer, groups, args):
    """Each epoch takes one optimizer step for the mean loss over all completions."""
    model.eval()  # Evaluation mode disables dropout while autograd remains enabled.
    total = sum(len(group.batch) for group in groups)
    epochs = []
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        metrics = {key: 0.0 for key in ("policy_loss", "clip_fraction", "active_clip_fraction",
                                        "approximate_behavior_kl")}
        ratio_min, ratio_max = math.inf, -math.inf
        for group in groups:
            for part in _scoring_slices(len(group.batch), args.score_microbatch):
                indices = torch.arange(part.start, part.stop, device=group.sequences.device)
                batch = group.batch.select(indices)
                result = ppo_actor_loss(seq_logprobs(model, group.sequences[part], group.mask[part]),
                                        batch, args.clip_coef)
                share = len(batch) / total
                (result.loss * share).backward()
                current = result.metrics()
                for key in metrics:
                    metrics[key] += current[key] * share
                ratio_min = min(ratio_min, current["ratio_min"])
                ratio_max = max(ratio_max, current["ratio_max"])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                              error_if_nonfinite=True)
        optimizer.step()
        epochs.append({"epoch": epoch + 1, **metrics, "ratio_min": ratio_min,
                       "ratio_max": ratio_max, "grad_norm_preclip": float(norm)})
    return epochs


def critic_update(critic, optimizer, groups, args):
    if critic is None:
        return {"critic_fit_steps": 0}
    fields = ("prompt_ids", "behavior_log_probs", "returns", "old_values", "advantages")
    batch = FrozenBanditBatch(**{field: torch.cat([getattr(group.batch, field) for group in groups])
                                for field in fields})
    initial = float(value_loss(critic(batch.prompt_ids), batch).detach())
    for _ in range(args.critic_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = value_loss(critic(batch.prompt_ids), batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("The critic loss is not finite.")
        loss.backward()
        optimizer.step()
    final = float(value_loss(critic(batch.prompt_ids), batch).detach())
    return {"critic_fit_steps": args.critic_steps, "critic_mse_before": initial,
            "critic_mse_after": final,
            "critic_values": critic(torch.arange(critic.values.num_embeddings,
                                                   device=batch.prompt_ids.device)).detach().cpu().tolist()}


def evaluate(model, tok, prompts, instances, args, eos_ids, step, split, completion_handle):
    """Fixed evaluation draws never consume the training generator's random state."""
    state = rng_state_snapshot()
    records = []
    try:
        seed_everything(args.eval_seed + (0 if split == "train" else 1))
        model.eval()
        for prompt, instance in zip(prompts, instances):
            plen = prompt.shape[1]
            # A bounded generation batch keeps final held-out evaluation within memory.
            for start in range(0, args.eval_samples, EVALUATION_BATCH_SIZE):
                count = min(EVALUATION_BATCH_SIZE, args.eval_samples - start)
                generated = sample_completions(model, tok, prompt, args, count)
                masks = completion_mask(generated, plen, eos_ids)
                for index, (sequence, mask) in enumerate(zip(generated, masks)):
                    record = _record(sequence, mask, plen,
                                     tok.decode(sequence[plen:], skip_special_tokens=True),
                                     instance, args, start + index, step, "eval", split)
                    records.append(record)
                    _append(completion_handle, record)
        return {"step": step, "phase": "eval", "split": split,
                **summarize_records(records),
                "per_instance": [{"instance_id": instance["instance_id"],
                                  **summarize_records([row for row in records
                                      if row["instance_id"] == instance["instance_id"]])}
                                 for instance in instances]}
    finally:
        rng_state_restore(state)


def _assert_finite_parameters(model):
    with torch.no_grad():
        if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise FloatingPointError("The final model has a nonfinite parameter.")


def configure_generation(model, tok, args):
    """Reset inherited logit processors and preserve the model's declared stop tokens."""
    from transformers import GenerationConfig
    original = getattr(model, "generation_config", None)
    stop_ids = getattr(original, "eos_token_id", None)
    if stop_ids is None:
        stop_ids = tok.eos_token_id
    stop_ids = ([stop_ids] if isinstance(stop_ids, int) else list(stop_ids or []))
    generation = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
        repetition_penalty=1.0, num_beams=1, max_new_tokens=args.max_new_tokens,
        eos_token_id=stop_ids or None, pad_token_id=tok.pad_token_id,
        bos_token_id=getattr(tok, "bos_token_id", None))
    model.generation_config = generation
    return sorted(set(stop_ids)), generation.to_dict()


def run(args):
    started = time.monotonic()
    dataset = task.load_manifest(args.dataset)
    validate_args(args, dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    device = pick_device(args.device)
    sources = source_fingerprints()
    cfg = {**vars(args), "protocol_version": PROTOCOL_VERSION,
           "dataset_sha256": dataset["sha256"], "kl_beta": 0.0,
           "evaluation_generation_batch_size": EVALUATION_BATCH_SIZE,
           "reference_penalty": "This experiment has no reference-policy penalty.",
           "sampling_distribution": "The unmodified model softmax defines both sampling and scoring.",
           "critic_type": "learned_prompt_value" if args.method == "ppo" else None,
           "critic_features": "Only the training prompt identity is available to the critic.",
           "evaluation_schedule": ("This timing smoke omits held-out evaluation." if args.skip_heldout else
               "Training prompts are evaluated initially and at fixed intervals. Held-out prompts are evaluated only after the last update."),
           "optimizer_steps": args.steps * args.epochs,
           "training_completions": args.steps * len(dataset["train"]) * args.group_size}
    cfg_hash = _digest({key: value for key, value in cfg.items() if key not in ("out", "dataset")})
    _write_json(out / "config.json", cfg)
    _write_json(out / "dataset.json", dataset)
    seed_everything(args.seed)
    tok, model, reference = build_models(args, device)
    if reference is not None:
        raise RuntimeError("This protocol must not allocate a reference model.")
    model.eval()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    eos_ids, generation = configure_generation(model, tok, args)
    prompts = {split: [chat_ids(tok, row["prompt"], device) for row in dataset[split]]
               for split in ("train", "heldout")}
    _write_json(out / "prompt_tokens.json", {split: [
        {"instance_id": row["instance_id"], "prompt_token_ids": ids[0].cpu().tolist()}
        for row, ids in zip(dataset[split], prompts[split])] for split in prompts})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    critic = PromptValueCritic(len(dataset["train"])).to(device) if args.method == "ppo" else None
    critic_optimizer = torch.optim.SGD(critic.parameters(), lr=args.critic_lr) if critic else None
    provenance = {"protocol_version": PROTOCOL_VERSION, "config_sha256": cfg_hash,
                  "dataset_sha256": dataset["sha256"], "source_sha256": _digest(sources),
                  "source_files": sources, "git_sha": git_sha(), "git_dirty": git_dirty(),
                  "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "python": sys.version.split()[0], "torch": torch.__version__,
                  "transformers": package_version("transformers"),
                  "numpy": np.__version__, "device": device, "device_name": device_name(device),
                  "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
                  "tokenizer_revision": getattr(tok, "init_kwargs", {}).get("_commit_hash"),
                  "effective_generation_config": generation, "completion_mask_stop_ids": eos_ids,
                  "hostname": socket.gethostname(), "platform": platform.platform(),
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                  "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                  "initial_param_checksum": param_checksum(model), "steps_completed": 0}
    _write_json(out / "provenance.json", provenance)
    print(json.dumps({"event": "start", "method": args.method, "config_sha256": cfg_hash}), flush=True)
    durations = []
    with (out / "metrics.jsonl").open("x") as metrics, (out / "completions.jsonl").open("x") as completions:
        _append(metrics, evaluate(model, tok, prompts["train"], dataset["train"], args,
                                  eos_ids, 0, "train", completions))
        for step in range(1, args.steps + 1):
            step_start = time.monotonic()
            groups = collect_rollout(model, tok, prompts["train"], dataset["train"],
                                     args, eos_ids, critic, step)
            for group in groups:
                for record in group.records:
                    _append(completions, record)
            actor_metrics = actor_update(model, optimizer, groups, args)
            critic_metrics = critic_update(critic, critic_optimizer, groups, args)
            duration = time.monotonic() - step_start
            durations.append(duration)
            records = [row for group in groups for row in group.records]
            record = {"step": step, "phase": "train", "split": "train", "s_per_step": duration,
                      **summarize_records(records), "actor_epochs": actor_metrics, **critic_metrics,
                      "all_violating_fraction": float(np.mean([
                          not any(row["useful_feasible"] for row in group.records) for group in groups])),
                      "zero_advantage_group_fraction": float(np.mean([
                          torch.count_nonzero(group.batch.advantages).item() == 0 for group in groups])),
                      "group_reward_std_mean": float(np.mean([
                          np.std([row["scaled_reward"] for row in group.records], ddof=1) for group in groups]))}
            _append(metrics, record)
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                print(json.dumps({key: record[key] for key in ("step", "useful_value_mean",
                                  "useful_feasible_rate", "s_per_step")}), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                _append(metrics, evaluate(model, tok, prompts["train"], dataset["train"], args,
                                          eos_ids, step, "train", completions))
            provenance["steps_completed"] = step
        if not args.skip_heldout:
            _append(metrics, evaluate(model, tok, prompts["heldout"], dataset["heldout"], args,
                                      eos_ids, args.steps, "heldout", completions))
        for handle in (metrics, completions):
            handle.flush()
            os.fsync(handle.fileno())
    _assert_finite_parameters(model)
    if critic is not None:
        _assert_finite_parameters(critic)
        _write_json(out / "critic.json", {"state_dict": {
            key: value.detach().cpu().tolist() for key, value in critic.state_dict().items()}})
    provenance.update({"end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "final_param_checksum": param_checksum(model),
                       "s_per_step_median": float(np.median(durations)),
                       "training_wall_seconds": float(sum(durations)),
                       "wall_seconds": float(time.monotonic() - started),
                       "artifact_sha256": {path.name: _file_digest(path) for path in out.iterdir()
                                           if path.name != "provenance.json"}})
    if torch.device(device).type == "cuda":
        provenance.update({"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                           "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
    if source_fingerprints() != sources:
        raise RuntimeError("Training source files changed during this run.")
    if provenance["steps_completed"] != args.steps:
        raise RuntimeError("The requested update budget was not completed.")
    _write_json(out / "provenance.json", provenance)
    _write_json(out / "DONE", {"status": "complete", "steps_completed": args.steps,
                               "provenance_sha256": _file_digest(out / "provenance.json")})
    print(json.dumps({"event": "done", "steps_completed": args.steps,
                      "wall_seconds": provenance["wall_seconds"]}), flush=True)
    return 0


def main(argv=None):
    return run(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
