#!/usr/bin/env python3
"""Matched-budget GRPO on the speed task, one arm per process.

Rebuilt from the earlier local script (task, prompts, parser, eval) and the LLM-scale
loop (batched completion log-probs, clipped surrogate, metrics stream), with the three
things the earlier runs lacked: every arm shares one budget, every step logs every group,
and the run writes its own provenance so the analysis can refuse an unmatched comparison.

The reward is a pure function of the parsed speed, so the logged speeds let the analysis
recompute every arm's advantages on every sampled group after the fact.

KL to a frozen copy of the base model is logged every step and is NOT in the loss by default
(beta = 0), matching provenance/objection_check.py. `--kl-beta` adds a per-token k3 penalty for the
KL ablation, and `--adv-variant` and `--group-filter` select the remedy arms. `--keep-schedule`
makes a yoked control, which keeps exactly the groups a committed schedule lists at each step. For
the ICML 2027 blocks (November 2026), `--adv-variant grpo_fixedref` with `--fixedref-c` shifts the
verifier component by a fixed reference in homogeneous-verdict groups, `--skip-zero-adv` keeps a
zero-advantage group out of the update, and a task that declares HELDOUT_PROMPTS is also evaluated on
those prompts. Every one of those flags defaults to the path the published runs took.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

import rewards as RW
import tasks
from speed_task import MODEL_ID, PROMPTS, parse_speed


# ------------------------------------------------------------------ args


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=RW.ARMS)
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--w-task", type=float, default=1.0,
                    help="weight on the task term; 0 reproduces the published shape where a "
                         "constant sits below feasibility and only feasible answers are graded")
    ap.add_argument("--vmax", type=float, required=True)
    ap.add_argument("--regime", default="", help="label only (A or B)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--batch-prompts", type=int, default=6, help="prompts per step; 6 = all of them")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-samples", type=int, default=16, help="samples per prompt at eval")
    ap.add_argument("--eval-seed", type=int, default=12345)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-new-tokens", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0, help="0 disables top-k; set explicitly so generation_config defaults do not leak in")
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="0 so a zero-advantage group produces exactly no parameter change")
    ap.add_argument("--clip-coef", type=float, default=0.2)
    # Memory, not mathematics. The step scores every group first and then calls backward once,
    # so the card holds one autograd graph per group. Each graph keeps a tensor the size of the
    # vocabulary distribution, about 2.4 GB at 128 new tokens, and six of those do not fit
    # beside the model, its gradients and the optimizer state on a 40 GB card.
    #
    # With this set, each group's share of the loss goes backward as soon as it is computed and
    # its graph is released. The gradient that reaches the optimizer is the same one, because
    # the sum of the per-group shares is the mean over all of them. The default of 0 keeps the
    # single-backward form the speed runs were produced with, so that path stays byte for byte
    # what it was.
    ap.add_argument("--backward-per-group", action="store_true",
                    help="release each group's graph as it is scored, instead of holding all "
                         "of them until one backward call; same gradient, far less memory")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--task", default="speed",
                    help="which constrained-generation task to train on; see tasks.py")
    ap.add_argument("--tiny", action="store_true",
                    help="random 2-layer model with the real tokenizer; tests only")
    ap.add_argument("--no-ref", action="store_true", help="skip the frozen reference (tests only)")
    # The remedy arms. Each default is the path the published runs took, and a test pins that.
    ap.add_argument("--adv-variant", default="grpo", choices=RW.TRAINER_ADV_VARIANTS,
                    help="advantage estimator; grpo is the frozen group-relative z-score, and "
                         "grpo_fixedref shifts it by a fixed reference in homogeneous-verdict groups")
    ap.add_argument("--group-filter", default="none", choices=RW.GROUP_FILTERS,
                    help="drop groups from the loss; a step whose every group is dropped takes no "
                         "optimizer step")
    ap.add_argument("--kl-beta", type=float, default=0.0,
                    help="weight on a per-token k3 KL penalty to the frozen reference, summed per "
                         "completion; 0 keeps the KL out of the loss")
    # This flag makes the yoked control. Its default reads no file and changes no computation.
    ap.add_argument("--keep-schedule", default=None,
                    help="id of an entry in configs/keep_schedules.json. The run keeps exactly "
                         "the listed prompts' groups at each step, and it takes an optimizer step "
                         "only when it keeps one")
    # ICML 2027 (November 2026). Both default to the path every earlier run took: the shift is read
    # only by grpo_fixedref, and the skip is off, so a zero-advantage group still enters the update.
    ap.add_argument("--fixedref-c", type=float, default=2.0,
                    help="the fixed-reference shift c, used only by --adv-variant grpo_fixedref: the "
                         "advantage is z - c in an all-violating group and z + c in an all-passing one")
    ap.add_argument("--skip-zero-adv", action="store_true",
                    help="a kept group whose advantages are all zero (within rewards.ZERO_ADV_TOL) "
                         "does not enter the update, and a step with no other group takes no "
                         "optimizer step; off by default, so AdamW's momentum still moves the weights")
    return ap


def check_args(ap: argparse.ArgumentParser, args):
    """Refuse flag combinations the experiment does not define, before anything is written.

    The KL ablation is run on the plain estimator only, so a remedy arm with a KL term would be a
    cell nobody designed. It also needs the frozen reference, which --no-ref removes.

    Returns the keep schedule as one frozenset of prompt indices per step, or None at the default."""
    if args.kl_beta < 0:
        ap.error("--kl-beta must not be negative")
    if args.kl_beta > 0:
        if args.no_ref:
            ap.error("--kl-beta > 0 needs the frozen reference, so it cannot run with --no-ref")
        # grpo_fixedref is refused here exactly as the other remedy variants are.
        if args.adv_variant != "grpo" or args.group_filter != "none":
            ap.error("--kl-beta > 0 is defined only with --adv-variant grpo and --group-filter none")
    return load_keep_schedule(ap, args)


# This file holds the yoked controls' schedules. make_keep_schedules.py writes it from the
# verifier-filter runs' kept counts, and the committed copy is a training source. The in-process
# tests point this constant at a temporary file.
KEEP_SCHEDULES_PATH = Path(__file__).resolve().parent / "configs" / "keep_schedules.json"
# An entry must share every one of these fields with the run's arguments before the run trusts it.
KEEP_SCHEDULE_FIELDS = ("task", "regime", "arm", "w", "w_task", "vmax", "seed", "steps", "batch_prompts")


def _read_keep_schedules(path) -> dict:
    return json.loads(Path(path).read_text())


def _same_field(want, got) -> bool:
    if isinstance(got, bool) or isinstance(want, bool):
        return want is got
    if isinstance(got, (int, float)):
        return isinstance(want, (int, float)) and float(want) == float(got)
    return want == got


def load_keep_schedule(ap: argparse.ArgumentParser, args):
    """Return the scheduled kept prompts per step, or None when --keep-schedule is unset.

    At the default this returns at once, so a plain run opens no file. With the flag set, the run
    is refused before anything is written unless the entry exists, describes this run, and lists one
    sorted set of distinct in-range prompt indices for every step. The schedule is defined only for
    the plain estimator without a filter or a KL term, because that is the arm the control copies."""
    if args.keep_schedule is None:
        return None
    if args.group_filter != "none" or args.adv_variant != "grpo" or args.kl_beta != 0:
        ap.error("--keep-schedule is defined only with --group-filter none, --adv-variant grpo "
                 "and --kl-beta 0")
    try:
        doc = _read_keep_schedules(KEEP_SCHEDULES_PATH)
    except (OSError, ValueError) as e:
        ap.error(f"--keep-schedule needs {KEEP_SCHEDULES_PATH.name}, and it could not be read: {e}")
    entries = doc.get("schedules") if isinstance(doc, dict) else None
    entry = entries.get(args.keep_schedule) if isinstance(entries, dict) else None
    if not isinstance(entry, dict):
        ap.error(f"--keep-schedule {args.keep_schedule!r} is not an entry of {KEEP_SCHEDULES_PATH.name}")
    wrong = [k for k in KEEP_SCHEDULE_FIELDS if not _same_field(entry.get(k), getattr(args, k))]
    if wrong:
        ap.error(f"--keep-schedule {args.keep_schedule!r} was written for another run: "
                 f"{', '.join(wrong)} differ")
    sets = entry.get("kept_sets")
    if not isinstance(sets, list) or len(sets) != args.steps:
        ap.error(f"--keep-schedule {args.keep_schedule!r} must list exactly {args.steps} kept sets")
    schedule = []
    for step, s in enumerate(sets, start=1):
        ok = (isinstance(s, list) and all(type(i) is int for i in s)
              and all(0 <= i < args.batch_prompts for i in s)
              and all(a < b for a, b in zip(s, s[1:])))
        if not ok:
            ap.error(f"--keep-schedule {args.keep_schedule!r}: the set at step {step} is not a sorted "
                     f"list of distinct prompt indices below {args.batch_prompts}")
        schedule.append(frozenset(s))
    return schedule


# The files a run's behaviour depends on, repo-relative. analyze.TRAINING_SOURCES must list the same
# files, and a test checks that. The digest below covers content, so it also describes a run made
# from a dirty tree, which the commit hash does not.
TRAINING_SOURCES = (
    "experiments/cancellation/train_speed_grpo.py",
    "experiments/cancellation/rewards.py",
    "experiments/cancellation/tasks.py",
    "experiments/cancellation/speed_task.py",
    "experiments/cancellation/capacity_task.py",
    "experiments/cancellation/make_capacity_instances.py",
    "experiments/cancellation/configs/base.yaml",
    "experiments/cancellation/configs/arms.yaml",
    "experiments/cancellation/configs/regimes.yaml",
    "experiments/cancellation/slurm/manifest.csv",
    "experiments/cancellation/slurm/manifest_capacity.csv",
    "experiments/cancellation/slurm/manifest_speed_remedy.csv",
    "experiments/cancellation/slurm/manifest_capacity_remedy.csv",
    "experiments/cancellation/slurm/manifest_capacity_kl.csv",
    "experiments/cancellation/slurm/manifest_fingerprint_speed.csv",
    "experiments/cancellation/configs/keep_schedules.json",
    "experiments/cancellation/slurm/manifest_speed_yoked.csv",
    "experiments/cancellation/slurm/manifest_capacity_yoked.csv",
    # Block A' of the ICML 2027 plan (added 2026-11-06): the bounded speed task and its manifest. They
    # sit at the end so the first fifteen entries stay the historical list.
    "experiments/cancellation/speed_bounded_task.py",
    "experiments/cancellation/slurm/manifest_speed_bounded.csv",
)
_SOURCE_PREFIX = "experiments/cancellation/"


def training_sources_digest(exp_dir=None) -> str:
    """sha256 over each training source's path and content hash, in the declared order.

    Paths resolve against the experiment directory, so the digest is the same in the repository and
    in a flattened copy. A missing file contributes a fixed marker and never raises."""
    base = Path(exp_dir) if exp_dir is not None else Path(__file__).resolve().parent
    h = hashlib.sha256()
    for rel in TRAINING_SOURCES:
        p = base / rel[len(_SOURCE_PREFIX):]
        h.update(rel.encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode() if p.is_file() else b"MISSING")
        h.update(b"\0")
    return h.hexdigest()


# ------------------------------------------------------------------ helpers


def _jsonable(x):
    """Parsed structures go into metrics.jsonl, so they must survive json.dumps."""
    if x is None or isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(_jsonable(v) for v in x)
    return float(x)


def pick_device(name: str):
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state_snapshot():
    import torch
    st = {"cpu": torch.get_rng_state(), "py": random.getstate(), "np": np.random.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        try:
            st["mps"] = torch.mps.get_rng_state()
        except Exception:  # older torch
            pass
    return st


def rng_state_restore(st):
    import torch
    torch.set_rng_state(st["cpu"])
    random.setstate(st["py"])
    np.random.set_state(st["np"])
    if "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])
    if "mps" in st:
        torch.mps.set_rng_state(st["mps"])


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                      stderr=subprocess.DEVNULL, text=True)
        return bool(out.strip())
    except Exception:
        return True


def param_checksum(model) -> str:
    import torch
    h = hashlib.sha256()
    with torch.no_grad():
        for p in model.parameters():
            h.update(p.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def device_name(dev: str) -> str:
    import torch
    if dev == "cuda":
        return torch.cuda.get_device_name(0)
    return dev


def build_models(args, dev: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.tiny:
        from transformers import Qwen2Config
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                          max_position_embeddings=512, tie_word_embeddings=True)
        cfg.pad_token_id = tok.pad_token_id
        cfg.eos_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_config(cfg)
        model = model.to(torch.float32)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    model = model.to(dev)
    ref = None
    if not args.no_ref:
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    return tok, model, ref


def chat_ids(tok, prompt: str, dev: str):
    s = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return tok(s, return_tensors="pt").input_ids.to(dev)


SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")


def sample_completions(model, tok, ids, args, n: int):
    """Sample n completions for one prompt.

    Every decoding knob is passed explicitly. Qwen2.5-0.5B-Instruct ships a generation_config of
    temperature 0.7, top_p 0.8, top_k 20 and repetition_penalty 1.1, so any caller that omits one
    silently samples from a different distribution than the trainer. That happened once: a
    calibration that omitted only repetition_penalty measured a mean selection size of 2.9 where
    the trainer produced 5.2, and it would have set both constraint regimes from the wrong
    distribution.
    """
    import torch
    with torch.no_grad():
        return model.generate(
            ids, do_sample=True, num_return_sequences=n, pad_token_id=tok.pad_token_id,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            repetition_penalty=args.repetition_penalty, max_new_tokens=args.max_new_tokens)


def completion_mask(seqs, plen: int, eos_ids: List[int]):
    """Mask of completion tokens to score: from plen up to and including the first EOS.

    seqs: (N, L) generated sequences sharing one prompt length plen.
    Returns a (N, L-1) bool mask in the shifted frame (position t predicts token t+1).
    """
    import torch
    N, L = seqs.shape
    comp = seqs[:, plen:]
    keep = torch.full((N,), comp.shape[1], dtype=torch.long, device=seqs.device)
    for e in eos_ids:
        hit = (comp == e)
        any_hit = hit.any(dim=1)
        first = torch.where(any_hit, hit.float().argmax(dim=1) + 1, keep)
        keep = torch.minimum(keep, first)
    pos = torch.arange(L - 1, device=seqs.device).unsqueeze(0)
    return (pos >= plen - 1) & (pos < plen - 1 + keep.unsqueeze(1))


def seq_logprobs(model, seqs, mask):
    """Summed log p(token) over the masked completion span. (N,) tensor."""
    import torch
    out = model(seqs)
    logits = out.logits[:, :-1, :]
    targets = seqs[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask).sum(dim=1)


def token_logprobs(model, seqs, mask):
    """Per-token log p(token) over the masked completion span, zero elsewhere. (N, L-1) tensor.

    Only the KL penalty needs per-token values. The default path never calls this, and keeps
    `seq_logprobs` exactly as the published runs used it."""
    import torch
    out = model(seqs)
    logits = out.logits[:, :-1, :]
    targets = seqs[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok_lp * mask


def policy_update(model, ref, opt, all_seqs, all_masks, all_adv, kept, args) -> dict:
    """One clipped-surrogate update over the kept groups, and the step's divergence statistics.

    With every group kept and --kl-beta 0 this is the published update, operation for operation:
    the mean of the per-completion terms goes backward once, or each group's share goes backward as
    it is scored under --backward-per-group, then the gradient is clipped and AdamW steps.

    A dropped group contributes nothing to the loss, and the mean runs over the kept completions
    only. When no group is kept there is no loss at all, so backward, clipping and the optimizer
    step are all skipped. Stepping anyway would still move the weights, because AdamW's first
    moment carries the previous nonzero gradient.

    `kept` is the list of groups that enter the update. With --skip-zero-adv the caller passes the
    filter decision with every zero-advantage group removed, so the mean of the loss runs over the
    completions of the groups that enter the update, and opt.step() is skipped when no group with a
    nonzero advantage remains. With the flag off the caller passes the filter decision itself and
    nothing about the update changes.

    The divergence to the reference is logged over every group, kept or not. A dropped group's
    policy log-probability is the no-grad one, which is the same forward pass on the same weights.
    """
    import torch
    opt.zero_grad()
    use_kl = args.kl_beta > 0
    n_terms = sum(int(a.numel()) for a, k in zip(all_adv, kept) if k)
    pg_terms, loss_terms, kl_terms = [], [], []
    k3_sum, k3_seq_all, running, running_pg = 0.0, [], 0.0, 0.0
    for gen, mask, a, keep in zip(all_seqs, all_masks, all_adv, kept):
        if use_kl:
            with torch.no_grad():
                tok_old = token_logprobs(model, gen, mask)
                tok_ref = token_logprobs(ref, gen, mask)
            logp_old, logp_ref = tok_old.sum(dim=1), tok_ref.sum(dim=1)
        else:
            with torch.no_grad():
                logp_old = seq_logprobs(model, gen, mask)
                logp_ref = seq_logprobs(ref, gen, mask) if ref is not None else None
        if not keep:
            if logp_ref is not None:
                kl_terms.append(logp_old - logp_ref)
            continue
        if use_kl:
            tok_new = token_logprobs(model, gen, mask)
            logp_new = tok_new.sum(dim=1)
        else:
            logp_new = seq_logprobs(model, gen, mask)
        ratio = (logp_new - logp_old).exp()
        unclipped = ratio * a
        clipped = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * a
        term = -torch.min(unclipped, clipped)
        full = term
        if use_kl:
            # k3 estimator per token, log(pi_ref / pi) = tok_ref - tok_new, summed per completion
            # exactly as the log-probabilities of the surrogate are.
            lr = (tok_ref - tok_new) * mask
            k3_seq = ((torch.exp(lr) - lr - 1.0) * mask).sum(dim=1)
            full = term + args.kl_beta * k3_seq
            k3_seq_all.append(k3_seq.detach())
            k3_sum += float(k3_seq.detach().sum())
        if args.backward_per_group:
            share = full.sum() / n_terms
            share.backward()
            running += float(share.detach())
            running_pg += float(term.detach().sum() / n_terms)
        else:
            pg_terms.append(term)
            loss_terms.append(full)
        if logp_ref is not None:
            kl_terms.append((logp_new.detach() - logp_ref))

    n_kept = sum(1 for k in kept if k)
    stats = {"n_kept_groups": n_kept, "opt_step_taken": n_kept > 0,
             "grad_norm_preclip": None, "pg_loss": None, "kl_loss": None, "kl_k3_tok_mean": None}
    if n_kept:
        if args.backward_per_group:
            pg_loss = torch.tensor(running_pg if use_kl else running)
            loss_val = running
        elif use_kl:
            pg_loss = torch.cat(pg_terms).mean()
            loss = torch.cat(loss_terms).mean()
            loss.backward()
            loss_val = float(loss.detach())
        else:
            pg_loss = torch.cat(pg_terms).mean()
            pg_loss.backward()
            loss_val = None
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        stats["grad_norm_preclip"] = float(total)
        stats["pg_loss"] = float(pg_loss.detach())
        if use_kl:
            stats["kl_loss"] = args.kl_beta * float(torch.cat(k3_seq_all).mean())
            stats["loss"] = loss_val
    n_tok = sum(int(m.sum()) for m in all_masks)
    if use_kl and n_tok:
        kept_tok = sum(int(m.sum()) for m, k in zip(all_masks, kept) if k)
        stats["kl_k3_tok_mean"] = k3_sum / kept_tok if kept_tok else None
    stats["kl_seq_mean"] = float(torch.cat(kl_terms).mean()) if kl_terms else None
    stats["kl_tok_mean"] = (float(torch.cat(kl_terms).sum()) / n_tok) if (kl_terms and n_tok) else None
    return stats


# ------------------------------------------------------------------ eval


def evaluate(model, tok, dev, args, eos_ids, spec=None):
    """Fixed-seed sampling eval. Restores the training RNG afterwards.

    The number and order of generate calls after seed_everything is the eval's identity, so
    the prompt list must be a fixed literal and is iterated in order, never shuffled."""
    import torch
    spec = spec if spec is not None else tasks.get(getattr(args, "task", "speed"))
    snap = rng_state_snapshot()
    seed_everything(args.eval_seed)
    model.eval()
    speeds: List[Optional[Any]] = []
    with torch.no_grad():
        plist = spec.PROMPTS[: args.batch_prompts] if args.batch_prompts else spec.PROMPTS
        for pi, p in enumerate(plist):
            parse_p = spec.parser_for(pi)
            ids = chat_ids(tok, p, dev)
            out = sample_completions(model, tok, ids, args, args.eval_samples)
            for seq in out:
                speeds.append(parse_p(tok.decode(seq[ids.shape[1]:], skip_special_tokens=True)))
        # ICML 2027 (November 2026). Held-out prompt wordings, when the task declares any. These
        # generate calls come after every training-prompt call, so the training-prompt samples above
        # are the same with or without them. The held-out prompts continue the prompt numbering, and
        # a task whose parser depends on the prompt index must accept those indices in parser_for.
        # Neither built-in task declares HELDOUT_PROMPTS, so this block is skipped on the frozen path.
        heldout_prompts = list(getattr(spec, "HELDOUT_PROMPTS", None) or [])
        heldout: List[Optional[Any]] = []
        for hj, p in enumerate(heldout_prompts):
            parse_p = spec.parser_for(len(spec.PROMPTS) + hj)
            ids = chat_ids(tok, p, dev)
            out = sample_completions(model, tok, ids, args, args.eval_samples)
            for seq in out:
                heldout.append(parse_p(tok.decode(seq[ids.shape[1]:], skip_special_tokens=True)))
    rng_state_restore(snap)
    model.train()
    present = [s for s in speeds if s is not None]
    n = len(speeds)
    viol = sum(1 for s in speeds if not spec.verdict(s, args.vmax))
    resid = [spec.residual(s, args.vmax) for s in present]
    ev = {
        "eval_n": n,
        "eval_violation_rate": viol / n,
        "eval_parse_fail_rate": (n - len(present)) / n,
        spec.EVAL_SCALAR_KEY: spec.summary_stat(speeds),
        "eval_mean_residual": float(np.mean(resid)) if resid else None,
        spec.EVAL_ITEMS_KEY: [s if s is None else _jsonable(s) for s in speeds],
    }
    if heldout_prompts:
        h_present = [s for s in heldout if s is not None]
        h_n = len(heldout)
        h_viol = sum(1 for s in heldout if not spec.verdict(s, args.vmax))
        h_resid = [spec.residual(s, args.vmax) for s in h_present]
        ev.update({
            "eval_heldout_n": h_n,
            "eval_heldout_violation_rate": h_viol / h_n,
            "eval_heldout_parse_fail_rate": (h_n - len(h_present)) / h_n,
            "eval_heldout_mean_residual": float(np.mean(h_resid)) if h_resid else None,
            "eval_heldout_items": [s if s is None else _jsonable(s) for s in heldout],
        })
    return ev


# ------------------------------------------------------------------ main


def main(argv=None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    schedule = check_args(ap, args)
    import torch

    dev = pick_device(args.device)
    TASK = tasks.get(args.task)          # pure dict read; consumes no randomness
    seed_everything(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    cfg = {k: v for k, v in vars(args).items()}
    cfg_for_hash = {k: v for k, v in cfg.items() if k not in ("out_dir",)}
    cfg_hash = hashlib.sha256(json.dumps(cfg_for_hash, sort_keys=True).encode()).hexdigest()[:16]
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))

    SPEC_ARG = None if TASK.name == "speed" else TASK   # speed keeps the frozen scalar path
    tok, model, ref = build_models(args, dev)
    eos_ids = sorted({i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None})
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    prompts = TASK.PROMPTS[: args.batch_prompts]
    K = args.group_size

    import transformers
    prov = {
        "git_sha": git_sha(), "git_dirty": git_dirty(), "config_hash": cfg_hash,
        "python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
        "numpy": np.__version__, "device": dev, "device_name": device_name(dev),
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps_completed": 0,
        "training_sources_digest": training_sources_digest(),
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True))
    print(json.dumps({"event": "start", **{k: prov[k] for k in ("git_sha", "device_name", "config_hash")},
                      "arm": args.arm, "w": args.w, "vmax": args.vmax, "seed": args.seed, "steps": args.steps}), flush=True)

    step_times = []
    model.train()
    ev = evaluate(model, tok, dev, args, eos_ids, TASK)
    metrics_path.open("a").write(json.dumps({"step": 0, "phase": "eval", **ev}) + "\n")

    for step in range(1, args.steps + 1):
        t0 = time.time()
        groups = []       # per prompt: dict of speeds, rewards, advantages
        all_seqs, all_masks, all_adv = [], [], []
        model.eval()
        with torch.no_grad():
            for pi, p in enumerate(prompts):
                parse_p = TASK.parser_for(pi)
                ids = chat_ids(tok, p, dev)
                plen = ids.shape[1]
                gen = sample_completions(model, tok, ids, args, K)
                texts = [tok.decode(s[plen:], skip_special_tokens=True) for s in gen]
                speeds = [parse_p(t) for t in texts]
                r = RW.group_rewards(speeds, args.arm, args.w, args.vmax, spec=SPEC_ARG, w_task=args.w_task)
                # Pure functions of the rewards and the design, so none of them draws randomness.
                # The verdict list is computed once and read by the filter, by the fixed-reference
                # variant and by the log.
                verdicts = RW.group_verdicts(speeds, args.vmax, spec=SPEC_ARG)
                if args.adv_variant == "grpo_fixedref":
                    anchor = None
                    a = RW.advantage_fixedref(r, verdicts, args.fixedref_c)   # float64, numpy
                else:
                    anchor = RW.reward_anchor(args.adv_variant, args.arm, args.w, args.vmax, spec=SPEC_ARG,
                                              w_task=args.w_task, instance=TASK.instance_for(pi))
                    a = RW.advantage_variant(r, args.adv_variant, anchor)   # float64, numpy
                kept = RW.keep_group(args.group_filter, r, verdicts)
                if schedule is not None:
                    # The yoked control keeps exactly the scheduled prompts. The schedule was fixed
                    # before the run from a filter run's kept counts, so this line reads no reward,
                    # advantage or verdict of this run, and it draws no randomness.
                    kept = pi in schedule[step - 1]
                mask = completion_mask(gen, plen, eos_ids)
                groups.append({TASK.ITEMS_KEY: [_jsonable(x) for x in speeds],
                               "rewards": r.tolist(), "adv": a.tolist(),
                               "all_violating": RW.group_is_all_violating(speeds, args.vmax, spec=SPEC_ARG),
                               "has_violator": RW.group_has_violator(speeds, args.vmax, spec=SPEC_ARG),
                               "kept": kept, "prompt_index": pi, "anchor": anchor,
                               "n_tokens": [int(n) for n in mask.sum(dim=1).tolist()],
                               "verdicts": verdicts, "zero_adv": RW.group_adv_is_zero(a)})
                all_seqs.append(gen)
                all_masks.append(mask)
                all_adv.append(torch.tensor(a, dtype=torch.float32, device=dev))

        # ---- old and reference log-probs (no grad), then the update ----
        # g["kept"] stays the filter decision. With --skip-zero-adv a kept group whose advantages are
        # all zero is removed from the list the update sees, and g["kept_for_update"] records that.
        # With the flag off the two lists are the same list of values.
        for g in groups:
            g["kept_for_update"] = bool(g["kept"] and not (args.skip_zero_adv and g["zero_adv"]))
        model.train()
        upd = policy_update(model, ref, opt, all_seqs, all_masks, all_adv,
                            [g["kept_for_update"] for g in groups], args)
        kl_seq_mean = upd["kl_seq_mean"]

        # ---- aggregates over the step ----
        flat_speeds = [s for g in groups for s in g[TASK.ITEMS_KEY]]
        flat_adv = np.array([x for g in groups for x in g["adv"]])
        viol_mask = np.array([not TASK.verdict(s, args.vmax) for s in flat_speeds])
        flat_present = [s for s in flat_speeds if s is not None]
        flat_resid = [TASK.residual(s, args.vmax) for s in flat_present]
        dt = time.time() - t0
        step_times.append(dt)
        rec = {
            "step": step, "phase": "train", "s_per_step": dt,
            "violation_rate": float(viol_mask.mean()),
            "parse_fail_rate": float(sum(s is None for s in flat_speeds) / len(flat_speeds)),
            "all_violating_fraction": float(np.mean([g["all_violating"] for g in groups])),
            TASK.SCALAR_KEY: TASK.summary_stat(flat_speeds),
            "mean_residual": float(np.mean(flat_resid)) if flat_resid else None,
            "reward_mean": float(np.mean([x for g in groups for x in g["rewards"]])),
            "group_reward_std_mean": float(np.mean([np.std(g["rewards"], ddof=1) for g in groups])),
            "violator_mean_adv": float(flat_adv[viol_mask].mean()) if viol_mask.any() else None,
            "frac_violators_pos_adv": float((flat_adv[viol_mask] > 0).mean()) if viol_mask.any() else None,
            "kl_seq_mean": kl_seq_mean, "pg_loss": upd["pg_loss"],
            # Remedy-arm fields. A plain run logs them too, so every run can be read the same way.
            "opt_step_taken": upd["opt_step_taken"], "grad_norm_preclip": upd["grad_norm_preclip"],
            "n_kept_groups": sum(1 for g in groups if g["kept"]),
            # ICML 2027 fields. Without --skip-zero-adv, n_update_groups equals n_kept_groups.
            "n_zero_adv_groups": sum(1 for g in groups if g["zero_adv"]),
            "n_update_groups": upd["n_kept_groups"],
            "n_mixed_groups": sum(1 for g in groups if g["has_violator"] and not g["all_violating"]),
            "n_all_feasible_groups": sum(1 for g in groups if not g["has_violator"]),
            "completion_tokens_mean": float(np.mean([n for g in groups for n in g["n_tokens"]])),
            "kl_tok_mean": upd["kl_tok_mean"],
            "kl_k3_tok_mean": upd["kl_k3_tok_mean"], "kl_loss": upd["kl_loss"],
            "groups": groups,
        }
        if schedule is not None:
            rec["keep_schedule_n"] = len(schedule[step - 1])
        metrics_path.open("a").write(json.dumps(rec) + "\n")
        if step % 10 == 0 or step == 1:
            print(json.dumps({k: rec[k] for k in ("step", "violation_rate", "all_violating_fraction",
                                                   "mean_residual", "kl_seq_mean", "pg_loss", "s_per_step")}), flush=True)
        if args.eval_every and (step % args.eval_every == 0 or step == args.steps):
            ev = evaluate(model, tok, dev, args, eos_ids, TASK)
            metrics_path.open("a").write(json.dumps({"step": step, "phase": "eval", **ev}) + "\n")
        prov["steps_completed"] = step

    prov["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov["final_param_checksum"] = param_checksum(model)
    prov["s_per_step_median"] = float(np.median(step_times)) if step_times else None
    prov["wall_seconds"] = float(sum(step_times))
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True))
    assert prov["steps_completed"] == args.steps, f"budget not met: {prov['steps_completed']} != {args.steps}"
    (out / "DONE").write_text("ok\n")
    print(json.dumps({"event": "done", "steps_completed": prov["steps_completed"],
                      "s_per_step_median": prov["s_per_step_median"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
