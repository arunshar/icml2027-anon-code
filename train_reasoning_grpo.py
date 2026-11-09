#!/usr/bin/env python3
"""This is the WO-I trainer, which runs mixed-reward GRPO on GSM8K with batched generation.

A step draws the six problems that the frozen schedule assigns to it and samples four completions
for each of them in one left-padded generate call. Each problem's padding is then removed on both
sides, so its four sequences share one prompt length and end at that problem's longest completion.
`seq_logprobs` then scores them at the same width and in the same way as the frozen trainer, with no
attention mask. The reward is the mixed reward of reasoning_task.py. The advantage and
the group filter are `advantage_variant` and `keep_group` of rewards.py, unchanged. The update is the
frozen clipped surrogate with one optimizer step per batch, scored two sequences at a time.

The trainer writes four files. metrics.jsonl holds every group and every evaluation, and it holds
no wall-clock field, so two runs with one seed on one device write the same bytes. completions.jsonl
holds every raw completion. timing.json holds the step times, the evaluation times and the peak GPU
memory, and it is the only file that the S rule and the memory rule of Section 2.3 read.
provenance.json holds the digests. No checkpoint is written.

On the cluster the trainer reads one row of a frozen manifest and refuses any row outside the
design. The explicit flags exist for tests and never produce an admissible run.

Design version 2 (ICML 2027, Block D' of the plan of November 2026) sits beside version 1. A
version-2 row carries a learning rate, a k3 KL penalty weight, the zero-advantage optimizer skip and
the fixed-reference shift c. With kl_beta > 0 the trainer loads a frozen copy of the same snapshot
and adds the per-token k3 penalty of train_speed_grpo.policy_update to each update completion. A
version-2 run also logs the per-token policy entropy, the sequence KL to the reference, the
zero-advantage group counts, and it stops training early when the job's time limit cannot hold
another step plus the final evaluations. A version-1 row takes the version-1 code path, which is
unchanged: no reference model is loaded, no new field enters metrics.jsonl, and the bytes it writes
are the bytes it wrote before version 2 existed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import reasoning_task as RT
import rewards as RW
import train_speed_grpo as TSG

DATASET_SCHEMA = "wo_i_reasoning_dataset_v1"
SELECTION_SCHEMA = "wo_i_reasoning_selection_v1"
HERE = Path(__file__).resolve().parent
# The training-source digest each design version expects, keyed by version. Version 1 is the value
# every frozen WO-I run of September 2026 recorded, and it stays as it is for those runs. Version 2
# is read from configs/reasoning_frozen_v2.json, which the experimenter writes by running
# package_reasoning_source.py once after every code change of the ICML 2027 blocks (November 2026)
# has landed. Until that file exists a version-2 row refuses to run. The training sources moved on
# 2026-11-06 (rewards.py and train_speed_grpo.py gained the grpo_fixedref variant, --fixedref-c,
# --skip-zero-adv and held-out prompt evaluation), so a new run of either version records the
# current digest in its provenance, and only a version-2 run checks it against the frozen file.
FROZEN_TRAINING_DIGEST = {
    RT.DESIGN_VERSION_V1: "6afdaa1fa50c1126c3278aa33887d4be4ec47e5283097bda081bfa563e3300aa",
    RT.DESIGN_VERSION_V2: None,
}
FROZEN_V2_PATH = HERE / "configs" / "reasoning_frozen_v2.json"
FROZEN_V2_SCHEMA = "wo_i_frozen_v2"
FROZEN_V2_HELP = ("the experimenter writes it by running package_reasoning_source.py --design-version 2 "
                  "--write-frozen-v2 once after every code change has landed")
# Seconds the early-finish guard keeps in hand for the bundle preflight, the final writes and the
# scheduler's own accounting, on top of one mean step and the final evaluations.
EARLY_FINISH_MARGIN_SECONDS = 120.0


EVAL_ITEM_FIELDS = ("pid", "sample", "n_tokens", "finished", "F", "strict", "lenient", "lenient_correct")


class Refused(RuntimeError):
    """A precondition failed before any training, so nothing was run."""


# ------------------------------------------------------------------ files


def load_json_checked(path, expected_sha256: Optional[str], schema: str) -> dict:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise Refused(f"{Path(path).name} has SHA-256 {digest}, and the run expects {expected_sha256}")
    data = json.loads(raw.decode("utf-8"))
    if data.get("schema") != schema:
        raise Refused(f"{Path(path).name} does not have the schema {schema}")
    return data


def candidates_by_pid(dataset: dict) -> Dict[str, dict]:
    return {c["pid"]: c for c in dataset["train_candidates"] + dataset["test_candidates"]}


def read_frozen_v2(path) -> dict:
    """Return the version-2 frozen digests, and refuse when the file is absent or malformed."""
    path = Path(path)
    if not path.is_file():
        raise Refused(f"a version-2 row needs {path}, which does not exist yet ({FROZEN_V2_HELP})")
    try:
        doc = json.loads(path.read_bytes().decode("utf-8"))
    except ValueError as exc:
        raise Refused(f"{path} is not valid JSON ({exc})")
    for key in ("training_sources_digest", "reasoning_sources_digest"):
        if not isinstance(doc.get(key), str) or len(doc[key]) != 64:
            raise Refused(f"{path} has no {key}")
    if doc.get("schema") != FROZEN_V2_SCHEMA:
        raise Refused(f"{path} does not have the schema {FROZEN_V2_SCHEMA}")
    return doc


def check_frozen_v2(path) -> dict:
    """Refuse a version-2 run whose current sources differ from the frozen version-2 digests."""
    doc = read_frozen_v2(path)
    training, reasoning = TSG.training_sources_digest(), RT.reasoning_sources_digest()
    if training != doc["training_sources_digest"]:
        raise Refused(f"the training sources hash to {training}, and {Path(path).name} freezes "
                      f"{doc['training_sources_digest']}")
    if reasoning != doc["reasoning_sources_digest"]:
        raise Refused(f"the reasoning sources hash to {reasoning}, and {Path(path).name} freezes "
                      f"{doc['reasoning_sources_digest']}")
    return doc


# ------------------------------------------------------------------ model


def model_snapshot(model_id: str, revision: str) -> Path:
    """Return the local snapshot directory of a pinned revision. It never contacts the Hub."""
    from huggingface_hub import snapshot_download
    try:
        path = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    except Exception as exc:  # the cache does not hold this revision
        raise Refused(f"{model_id} at revision {revision} is not in the local cache ({type(exc).__name__})")
    if path.name != revision:
        raise Refused(f"the cached snapshot {path.name} is not the pinned revision {revision}")
    return path


def load_model(model_id: str, revision: str, dev: str, tiny: bool = False):
    """Return the tokenizer and a float32 model. The tiny path keeps the real tokenizer and chat template
    of the pinned 0.5B snapshot and builds a random two-layer model, as the frozen trainer does."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap = model_snapshot(model_id, revision)
    tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tiny:
        from transformers import Qwen2Config
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                          max_position_embeddings=2048, tie_word_embeddings=True)
        cfg.pad_token_id = tok.pad_token_id
        cfg.eos_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_config(cfg).to(torch.float32)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(str(snap), dtype=torch.float32, local_files_only=True)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(str(snap), torch_dtype=torch.float32, local_files_only=True)
    files = {}
    for name in ("tokenizer.json", "config.json"):
        p = snap / name
        files[name] = RT.file_sha256(p) if p.is_file() else None
    return tok, model.to(dev), {"snapshot": snap.name, "file_sha256": files}


def frozen_reference(model):
    """Return a frozen copy of the policy, in eval mode with no gradient, as train_speed_grpo.build_models does.

    Only a version-2 run with kl_beta > 0 calls this. It doubles the weight memory: for the 0.5B
    float32 policy the copy adds about 2 GB."""
    import copy
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def eos_ids_of(tok) -> List[int]:
    return sorted({i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None})


def encode_prompt(tok, prompt: str) -> List[int]:
    """Return the token ids of the chat-formatted prompt, the same ids as train_speed_grpo.chat_ids."""
    s = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                add_generation_prompt=True)
    return list(tok(s).input_ids)


@dataclass
class Sampler:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0


def left_pad(prompt_ids: Sequence[Sequence[int]], pad_id: int, dev: str):
    import torch
    width = max(len(ids) for ids in prompt_ids)
    input_ids = torch.full((len(prompt_ids), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(prompt_ids), width), dtype=torch.long)
    for i, ids in enumerate(prompt_ids):
        input_ids[i, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
        attention[i, width - len(ids):] = 1
    return input_ids.to(dev), attention.to(dev), width


def group_width(seqs, plen: int, eos_ids: List[int]) -> int:
    """Return the prompt length plus the longest completion of one group.

    A completion's length counts its tokens up to and including the first end token, or every
    generated token when no end token came. The frozen trainer generated each group alone, and the
    generation config of the pinned 0.5B snapshot stops on the same two end ids, so the frozen
    trainer's groups had this width."""
    return plen + int(TSG.completion_mask(seqs, plen, eos_ids).sum(dim=1).max())


def trim_group(seqs, plen: int, eos_ids: List[int]):
    """Cut a group's sequences to group_width, which drops only columns after every row's first end token.

    Every scored position lies inside the kept columns. The model is causal and scores with no
    attention mask, so a dropped column does not enter the log-probability of any kept token. The
    token counts, masks, rewards and advantages are therefore unchanged. The forward pass runs on a
    narrower tensor, so the log-probabilities and gradients can still differ by float32 rounding."""
    return seqs[:, :group_width(seqs, plen, eos_ids)].contiguous()


def batched_generate(model, tok, prompt_ids: Sequence[Sequence[int]], n: int, sampler: Sampler,
                     max_new_tokens: int, dev: str, return_logits: bool = False):
    """Sample n completions for every prompt in one left-padded call.

    Returns, per prompt, its n sequences and its prompt length. Each group loses its left padding,
    and it also loses the right padding that the batch adds after the group's longest completion.
    Each group is therefore scored at its own width, as in the frozen trainer, and not at the width
    of the longest completion in the batch. Every sampling knob is passed explicitly, because the
    Qwen generation_config carries other values (see train_speed_grpo.sample_completions)."""
    import torch
    input_ids, attention, width = left_pad(prompt_ids, tok.pad_token_id, dev)
    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=attention, do_sample=True, num_return_sequences=n,
                             pad_token_id=tok.pad_token_id, temperature=sampler.temperature,
                             top_p=sampler.top_p, top_k=sampler.top_k,
                             repetition_penalty=sampler.repetition_penalty, max_new_tokens=max_new_tokens,
                             return_dict_in_generate=True, output_logits=return_logits)
    seqs = out.sequences
    eos_ids = eos_ids_of(tok)
    groups = []
    for i, ids in enumerate(prompt_ids):
        rows = seqs[i * n:(i + 1) * n]
        stripped = rows[:, width - len(ids):]
        if not bool((stripped[:, :len(ids)] == torch.tensor(ids, device=stripped.device)).all()):
            raise RuntimeError("padding removal did not recover the prompt")
        groups.append((trim_group(stripped, len(ids), eos_ids), len(ids)))
    if return_logits:
        return groups, out, width
    return groups


def completion_records(tok, seqs, plen: int, eos_ids: List[int]):
    """Return, per completion, its token count through the first end token, whether an end token
    came, its token ids and its decoded text. It also returns the scoring mask."""
    mask = TSG.completion_mask(seqs, plen, eos_ids)
    n_tokens = [int(x) for x in mask.sum(dim=1).tolist()]
    out = []
    for i, n in enumerate(n_tokens):
        ids = seqs[i, plen:plen + n].tolist()
        finished = bool(ids) and ids[-1] in eos_ids
        out.append({"n_tokens": n, "finished": finished, "token_ids": ids,
                    "text": tok.decode(ids, skip_special_tokens=True)})
    return out, mask


def score_texts(records: List[dict], reference) -> None:
    """Add both parses and the strict verdict F to each completion record."""
    for r in records:
        strict = RT.strict_parse(r["text"])
        r["strict"] = RT.fraction_str(strict)
        r["lenient"] = RT.fraction_str(RT.lenient_parse(r["text"]))
        r["F"] = 0 if (strict is not None and strict == reference) else 1


# ------------------------------------------------------------------ update


@dataclass
class Group:
    pid: str
    seqs: object
    mask: object
    adv: object      # the advantages are a float32 tensor on the device
    kept: bool
    # Version 2 only. zero_adv is True when every advantage is within rewards.ZERO_ADV_TOL of zero,
    # and kept_for_update is the group's share of the update: kept, and not zero_adv when the skip
    # is on. Version 1 never reads either field, and the defaults keep its constructor calls as they were.
    zero_adv: bool = False
    kept_for_update: bool = True


def scoring_slices(size: int, micro_batch: int):
    return [slice(s, min(s + micro_batch, size)) for s in range(0, size, micro_batch)]


def token_logprobs_and_entropy(model, seqs, mask):
    """Return the per-token log p(token) and the per-token entropy over the masked span, from one forward pass.

    The log-probabilities are the ones train_speed_grpo.token_logprobs returns, computed the same way
    (float32 log_softmax of the shifted logits). The entropy of each next-token distribution is
    -(softmax * log_softmax).sum(-1), masked to the completion tokens, so a version-2 run gets its
    entropy from the no-grad pass that already scores the old log-probabilities and pays no extra
    forward pass for it. Both tensors have shape (N, L-1) and are zero outside the mask."""
    import torch
    out = model(seqs)
    logits = out.logits[:, :-1, :]
    targets = seqs[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    entropy = -(logp.exp() * logp).sum(dim=-1)
    return tok_lp * mask, entropy * mask


def accumulate_gradients(model, groups: Sequence[Group], micro_batch: int, clip_coef: float) -> dict:
    """Run backward on the clipped surrogate of every kept group, micro_batch sequences at a time.

    Each micro-batch holds completions of one problem, so its sequences share one prompt length.
    The shares sum to the mean over every kept completion, which is the loss of the frozen
    policy_update, so the gradient is the same up to summation order."""
    import torch
    kept = [g for g in groups if g.kept]
    n_terms = sum(int(g.adv.numel()) for g in kept)
    pg = 0.0
    for g in kept:
        for part in scoring_slices(int(g.adv.numel()), micro_batch):
            seqs, mask, a = g.seqs[part], g.mask[part], g.adv[part]
            with torch.no_grad():
                old = TSG.seq_logprobs(model, seqs, mask)
            new = TSG.seq_logprobs(model, seqs, mask)
            ratio = (new - old).exp()
            term = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * a)
            share = term.sum() / n_terms
            share.backward()
            pg += float(share.detach())
    return {"n_kept_groups": len(kept), "opt_step_taken": bool(kept), "pg_loss": pg if kept else None}


def micro_batched_update(model, opt, groups: Sequence[Group], micro_batch: int, clip_coef: float,
                         grad_clip: float) -> dict:
    """Take one optimizer step over the kept groups. With no kept group there is no loss and no step,
    because AdamW would still move the weights on its stored moments (see policy_update)."""
    import torch
    opt.zero_grad()
    stats = accumulate_gradients(model, groups, micro_batch, clip_coef)
    stats["grad_norm_preclip"] = None
    if stats["opt_step_taken"]:
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        stats["grad_norm_preclip"] = float(total)
    return stats


# ------------------------------------------------------------------ update, design version 2


def accumulate_gradients_v2(model, ref, groups: Sequence[Group], micro_batch: int, clip_coef: float,
                            kl_beta: float) -> dict:
    """Run backward on the clipped surrogate, plus the k3 KL penalty when kl_beta > 0, over the update groups.

    This is accumulate_gradients with three additions, each the version-2 counterpart of something in
    train_speed_grpo.policy_update. The update groups are the ones with kept_for_update, so a
    zero-advantage group the skip removes contributes nothing and the mean runs over the completions
    that enter the update. With kl_beta > 0 the per-token k3 estimator of log(pi_ref / pi), which is
    exp(x) - x - 1 for x = log pi_ref - log pi, is summed per completion exactly as the
    log-probabilities of the surrogate are, weighted by kl_beta and added to that completion's term.
    The no-grad pass that scores the old log-probabilities also returns the per-token entropy, and
    it runs over every group, kept or not, as does the reference pass, so entropy_tok_mean and
    kl_seq_mean describe the whole step's completions.

    kl_seq_mean is the mean over all completions of the sum over completion tokens of log pi minus
    log pi_ref, under the current (old) weights. kl_k3_tok_mean is the k3 sum over the update
    completions divided by their token count, and kl_loss is kl_beta times the mean k3 sum over the
    update completions, as the speed trainer logs them."""
    import torch
    update = [g for g in groups if g.kept_for_update]
    n_terms = sum(int(g.adv.numel()) for g in update)
    use_kl = kl_beta > 0
    if use_kl and ref is None:
        raise RuntimeError("kl_beta > 0 needs the frozen reference")
    pg, loss, k3_sum, k3_seq_all, kl_terms = 0.0, 0.0, 0.0, [], []
    ent_sum, n_tok, kept_tok = 0.0, 0, 0
    for g in groups:
        for part in scoring_slices(int(g.adv.numel()), micro_batch):
            seqs, mask, a = g.seqs[part], g.mask[part], g.adv[part]
            with torch.no_grad():
                tok_old, ent = token_logprobs_and_entropy(model, seqs, mask)
                tok_ref = TSG.token_logprobs(ref, seqs, mask) if ref is not None else None
            old = tok_old.sum(dim=1)
            ent_sum += float(ent.sum())
            n_tok += int(mask.sum())
            if tok_ref is not None:
                kl_terms.append(old - tok_ref.sum(dim=1))
            if not g.kept_for_update:
                continue
            kept_tok += int(mask.sum())
            tok_new = TSG.token_logprobs(model, seqs, mask)
            new = tok_new.sum(dim=1)
            ratio = (new - old).exp()
            term = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * a)
            full = term
            if use_kl:
                x = (tok_ref - tok_new) * mask
                k3_seq = ((torch.exp(x) - x - 1.0) * mask).sum(dim=1)
                full = term + kl_beta * k3_seq
                k3_seq_all.append(k3_seq.detach())
                k3_sum += float(k3_seq.detach().sum())
            share = full.sum() / n_terms
            share.backward()
            pg += float(term.detach().sum() / n_terms)
            loss += float(share.detach())
    stats = {"n_kept_groups": sum(1 for g in groups if g.kept),
             "n_zero_adv_groups": sum(1 for g in groups if g.zero_adv),
             "n_update_groups": len(update), "opt_step_taken": bool(update),
             "pg_loss": pg if update else None, "loss": loss if update else None,
             "kl_loss": None, "kl_k3_tok_mean": None, "kl_seq_mean": None,
             "entropy_tok_mean": ent_sum / n_tok if n_tok else None}
    if use_kl:
        if update:
            stats["kl_loss"] = kl_beta * float(torch.cat(k3_seq_all).mean())
            stats["kl_k3_tok_mean"] = k3_sum / kept_tok if kept_tok else None
        stats["kl_seq_mean"] = float(torch.cat(kl_terms).mean()) if kl_terms else None
    return stats


def micro_batched_update_v2(model, ref, opt, groups: Sequence[Group], micro_batch: int, clip_coef: float,
                            grad_clip: float, kl_beta: float) -> dict:
    """Take one optimizer step over the update groups, and none when no group enters the update.

    The skip of version 2 is in the groups' kept_for_update flags, which the caller sets. A step with
    no update group takes no optimizer step, so AdamW's stored moments do not move the weights, which
    the WO-I collapse investigation of 25 September 2026 identified as a driver of the drift."""
    import torch
    opt.zero_grad()
    stats = accumulate_gradients_v2(model, ref, groups, micro_batch, clip_coef, kl_beta)
    stats["grad_norm_preclip"] = None
    if stats["opt_step_taken"]:
        total = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        stats["grad_norm_preclip"] = float(total)
    return stats


def group_advantages(r, fails: Sequence[int], design: dict):
    """Return one group's advantages under the design's estimator.

    Version 1 calls rewards.advantage_variant, unchanged. The grpo_fixedref variant of the restored
    arm (version 2) reads the verdict vector as well, through rewards.advantage_fixedref."""
    if design["adv_variant"] == "grpo_fixedref":
        return RW.advantage_fixedref(r, [f == 0 for f in fails], design["fixedref_c"])
    return RW.advantage_variant(r, design["adv_variant"])


def early_finish_needed_seconds(step_seconds: Sequence[float], full_eval_seconds: float,
                                mid_eval_scheduled: bool) -> Optional[float]:
    """Return the seconds the next step must still fit, or None before the first step is timed.

    The need is one mean step, the final evaluations, and one more full-evaluation time when a
    middle evaluation is scheduled after the step. The middle evaluation is smaller than a full one
    (one hard-set sample against four hard and two random), so the bound is conservative."""
    if not step_seconds:
        return None
    need = float(np.mean(step_seconds)) + full_eval_seconds
    if mid_eval_scheduled:
        need += full_eval_seconds
    return need


# ------------------------------------------------------------------ evaluation


def evaluate(model, tok, dev, sets, cands: Dict[str, dict], eos_ids, L: int, sampler: Sampler,
             eval_seed: int, chunk_sequences: int, short_threshold: Optional[float], step: int,
             completions_fh, timing: dict) -> List[dict]:
    """Run the fixed-seed held-out evaluation, and restore the training random state afterwards.

    sets is a list of (name, pids, samples per problem), generated in that order, with the problems
    of each set batched in their listed order."""
    snap = TSG.rng_state_snapshot()
    TSG.seed_everything(eval_seed)
    model.eval()
    results = []
    for name, pids, k in sets:
        t0 = time.time()
        per_call = max(1, chunk_sequences // k)
        items = []
        for start in range(0, len(pids), per_call):
            chunk = pids[start:start + per_call]
            groups = batched_generate(model, tok, [encode_prompt(tok, cands[p]["prompt"]) for p in chunk],
                                      k, sampler, L, dev)
            for pid, (seqs, plen) in zip(chunk, groups):
                recs, _ = completion_records(tok, seqs, plen, eos_ids)
                reference = cands[pid]["reference"]
                score_texts(recs, RT.parse_fraction_str(reference))
                for j, r in enumerate(recs):
                    completions_fh.write(json.dumps({"step": step, "phase": "eval", "set": name, "pid": pid,
                                                     "sample": j, "token_ids": r["token_ids"],
                                                     "text": r["text"]}) + "\n")
                    items.append([pid, j, r["n_tokens"], r["finished"], r["F"], r["strict"], r["lenient"],
                                  int(r["lenient"] is not None and r["lenient"] == reference)])
        timing.setdefault("eval_seconds", []).append({"step": step, "set": name, "samples": k,
                                                      "seconds": time.time() - t0})
        results.append(summarize_eval(name, k, items, short_threshold, step))
    TSG.rng_state_restore(snap)
    model.train()
    return results


def summarize_eval(name: str, k: int, items: List[list], short_threshold: Optional[float], step: int) -> dict:
    """Aggregate one held-out evaluation. Each item follows EVAL_ITEM_FIELDS.

    Accuracy is the mean over problems of the share of samples with F = 0. The lenient accuracy
    uses the last number in the completion and is a parser-sensitivity check only."""
    by_pid: Dict[str, List[list]] = {}
    for it in items:
        by_pid.setdefault(it[0], []).append(it)
    n = len(items)

    def per_problem(index, correct_value):
        return float(np.mean([np.mean([it[index] == correct_value for it in v]) for v in by_pid.values()]))

    return {
        "step": step, "phase": "eval", "set": name, "samples": k, "n_problems": len(by_pid), "n": n,
        "accuracy": per_problem(4, 0) if by_pid else None,
        "lenient_accuracy": per_problem(7, 1) if by_pid else None,
        "strict_parse_fail_rate": sum(1 for it in items if it[5] is None) / n if n else None,
        "mean_completion_tokens": float(np.mean([it[2] for it in items])) if n else None,
        "truncation_rate": sum(1 for it in items if not it[3]) / n if n else None,
        "short_incorrect_rate": (sum(1 for it in items if it[4] == 1 and it[2] < short_threshold) / n
                                 if (n and short_threshold is not None) else None),
        "item_fields": list(EVAL_ITEM_FIELDS),
        "items": items,
    }


# ------------------------------------------------------------------ args


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="the frozen arm manifest, which sets every design field together with --row")
    ap.add_argument("--row", type=int)
    ap.add_argument("--expect-stage", type=int, default=None,
                    help="with --manifest, refuse a row whose stage differs from the stage being released")
    ap.add_argument("--out-root", help="with --manifest, the run directory is <out-root>/<array id>/<name>")
    ap.add_argument("--out-dir", help="explicit run directory, for tests")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--kind", choices=("smoke", "study", "pilot"), default="study")
    ap.add_argument("--size", choices=RT.SIZES, default="0.5b")
    ap.add_argument("--arm", choices=tuple(RT.ARMS), default="plain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--L", type=int, default=384)
    ap.add_argument("--micro-batch", type=int, default=RT.MICRO_BATCH)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--dataset-sha256", default=None)
    ap.add_argument("--selection-sha256", default=None)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip-coef", type=float, default=0.2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-seed", type=int, default=RT.EVAL_SEED)
    ap.add_argument("--eval-chunk-sequences", type=int, default=RT.EVAL_CHUNK_SEQUENCES)
    ap.add_argument("--mid-eval-every", type=int, default=None,
                    help="default 50 for a study run and 10 for a smoke")
    ap.add_argument("--eval-limit", type=int, default=None, help="caps each held-out set, for tests only")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tiny", action="store_true", help="a random 2-layer model with the real tokenizer, for tests only")
    ap.add_argument("--require-gpu-name", default=None)
    ap.add_argument("--time-limit-minutes", type=int, default=None)
    # Design version 2 (ICML 2027, November 2026). Every default is the version-1 path. In manifest mode
    # the row sets each of these and the flags must stay at their defaults.
    ap.add_argument("--design-version", type=int, choices=RT.DESIGN_VERSIONS, default=RT.DESIGN_VERSION_V1,
                    help="1 is the frozen WO-I design, 2 adds lr, kl_beta, skip_zero_adv and fixedref_c")
    ap.add_argument("--kl-beta", type=float, default=0.0,
                    help="weight on the per-token k3 KL penalty to a frozen copy of the snapshot, for version 2 only")
    ap.add_argument("--skip-zero-adv", action="store_true",
                    help="a kept group whose advantages are all zero does not enter the update, for version 2 only")
    ap.add_argument("--fixedref-c", type=float, default=None,
                    help="the fixed-reference shift of the restored arm, which defaults to 2.0 for restored and 0 otherwise")
    ap.add_argument("--frozen-v2", default=str(FROZEN_V2_PATH),
                    help="the frozen version-2 digests a version-2 run checks itself against")
    ap.add_argument("--early-finish-margin-seconds", type=float, default=EARLY_FINISH_MARGIN_SECONDS,
                    help="seconds the early-finish guard keeps in hand, for version 2 with --time-limit-minutes only")
    return ap


def resolve_design(args) -> dict:
    """Return every design field, taken from the manifest row when one is given."""
    if args.manifest:
        if args.row is None or not args.out_root:
            raise Refused("--manifest needs --row and --out-root")
        row = RT.validate_arm_row(RT.read_manifest_row(args.manifest, args.row))
        if args.expect_stage is None or row["stage"] != args.expect_stage:
            raise Refused(f"row {args.row} is a stage {row['stage']} row, and stage {args.expect_stage} is released")
        spec = RT.ARMS[row["arm"]]
        design = {"mode": "manifest", "manifest_row": {k: str(v) for k, v in row.items()},
                  "task_id": row["task_id"], "kind": row["kind"], "stage": row["stage"], "size": row["size"],
                  "arm": row["arm"], "seed": row["seed"], "steps": row["steps"], "L": row["L"],
                  "micro_batch": row["micro_batch"], "model_id": row["model_id"],
                  "model_revision": row["model_revision"], "dataset_sha256": row["dataset_sha256"],
                  "selection_sha256": row["selection_sha256"]}
    else:
        spec = RT.ARMS[args.arm]
        model = RT.MODELS[args.size]
        design = {"mode": "explicit", "manifest_row": None, "task_id": None, "kind": args.kind, "stage": None,
                  "size": args.size, "arm": args.arm, "seed": args.seed, "steps": args.steps, "L": args.L,
                  "micro_batch": args.micro_batch, "model_id": args.model_id or model["model_id"],
                  "model_revision": args.model_revision or model["revision"],
                  "dataset_sha256": args.dataset_sha256, "selection_sha256": args.selection_sha256}
    design.update({"form": spec["form"], "w": RT.W, "w_task": float(spec["w_task"]),
                   "adv_variant": spec["adv_variant"], "group_filter": spec["group_filter"],
                   "group_size": RT.GROUP_SIZE, "groups_per_step": RT.GROUPS_PER_STEP})
    if args.mid_eval_every is not None:
        design["mid_eval_every"] = args.mid_eval_every
    else:
        design["mid_eval_every"] = RT.SMOKE_MID_EVAL_EVERY if design["kind"] == "smoke" else RT.EVAL_EVERY
    return _resolve_version(args, design, row if args.manifest else None)


def _resolve_version(args, design: dict, row: Optional[dict]) -> dict:
    """Attach the version-2 levers to a version-2 design, and refuse them on a version-1 design.

    A version-1 design gains no key here, so its config.json is the file it always was. In manifest
    mode the row's design version decides, and the version-2 flags must be at their defaults because
    the row sets every lever. In explicit mode --design-version decides."""
    ap_defaults = {"lr": 1e-5, "kl_beta": 0.0, "skip_zero_adv": False, "fixedref_c": None}
    if row is not None:
        version = row["design_version"]
        moved = [k for k, d in ap_defaults.items() if getattr(args, k) != d]
        if moved:
            raise Refused(f"in manifest mode the row sets {', '.join(moved)}, so the flags must stay at their defaults")
    else:
        version = args.design_version
    if version == RT.DESIGN_VERSION_V1:
        if args.kl_beta != 0.0 or args.skip_zero_adv or args.fixedref_c is not None:
            raise Refused("--kl-beta, --skip-zero-adv and --fixedref-c belong to design version 2")
        if design["arm"] == "restored" or design["kind"] == "pilot":
            raise Refused("the restored arm and the pilot kind belong to design version 2")
        return design
    if design["arm"] == "restored":
        want_c = RT.FIXEDREF_C
    else:
        want_c = 0.0
    if row is not None:
        levers = {"lr": row["lr"], "kl_beta": row["kl_beta"], "skip_zero_adv": bool(row["skip_zero_adv"]),
                  "fixedref_c": row["fixedref_c"]}
    else:
        if design["kind"] == "smoke":
            raise Refused("design version 2 has no smoke kind")
        if args.kl_beta < 0:
            raise Refused("--kl-beta must not be negative")
        c = want_c if args.fixedref_c is None else args.fixedref_c
        levers = {"lr": args.lr, "kl_beta": args.kl_beta, "skip_zero_adv": bool(args.skip_zero_adv),
                  "fixedref_c": c}
    if design["adv_variant"] != "grpo_fixedref" and levers["fixedref_c"] != 0.0:
        raise Refused("fixedref_c is read by the restored arm only, so it must be 0 elsewhere")
    design.update({"design_version": RT.DESIGN_VERSION_V2, **levers})
    return design


def check_time_limit(kind: str, minutes: Optional[int]) -> None:
    if minutes is None:
        return
    need = RT.MIN_MINUTES[kind]
    if minutes < need:
        raise Refused(f"a {kind} task needs at least {need} minutes, and this job has {minutes}")


def run_dir_for(args, design) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    array = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    name = f"{design['task_id']}_{design['kind']}_{design['size']}_{design['arm']}_s{design['seed']}"
    return Path(args.out_root) / array / name


# ------------------------------------------------------------------ main


def main(argv=None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    try:
        return _run(args)
    except (Refused, ValueError) as exc:
        print(f"REFUSED: {exc}. Nothing was trained.", file=sys.stderr, flush=True)
        return 2


def _run(args) -> int:
    import torch
    t_process = time.time()
    design = resolve_design(args)
    v2 = design.get("design_version") == RT.DESIGN_VERSION_V2
    frozen_v2 = check_frozen_v2(args.frozen_v2) if v2 else None
    check_time_limit(design["kind"], args.time_limit_minutes)
    dataset = load_json_checked(args.dataset, design["dataset_sha256"], DATASET_SCHEMA)
    selection = load_json_checked(args.selection, design["selection_sha256"], SELECTION_SCHEMA)
    dataset_sha = RT.file_sha256(args.dataset)
    if selection["candidates_sha256"] != dataset_sha:
        raise Refused("the selection was not made from this dataset file")
    size_sel = selection["sizes"].get(design["size"], {})
    if size_sel.get("status") != "ok":
        raise Refused(f"the {design['size']} size is not selected for training")
    if int(selection["L"]) != design["L"] and design["mode"] == "manifest":
        raise Refused("the row's L differs from the selection's L")
    cands = candidates_by_pid(dataset)
    schedule = size_sel["schedule"]
    if len(schedule) < design["steps"]:
        raise Refused("the prompt schedule is shorter than the step count")
    hard, rand = list(size_sel["hard"]), list(selection["random"])
    if args.eval_limit is not None:
        hard, rand = hard[:args.eval_limit], rand[:args.eval_limit]
    short_threshold = size_sel.get("short_threshold")

    dev = TSG.pick_device(args.device)
    if args.require_gpu_name is not None:
        name = TSG.device_name(dev)
        if name != args.require_gpu_name:
            raise Refused(f"the device is {name!r}, and the design needs {args.require_gpu_name!r}")

    out = run_dir_for(args, design)
    if out.exists() and any(out.iterdir()):
        raise Refused(f"the run directory {out} already holds files")

    # The model loads before the run directory exists, so a missing model leaves no directory behind.
    TSG.seed_everything(design["seed"])
    t_load = time.time()
    tiny_model = RT.MODELS["0.5b"]
    tok, model, model_info = load_model(tiny_model["model_id"] if args.tiny else design["model_id"],
                                        tiny_model["revision"] if args.tiny else design["model_revision"],
                                        dev, tiny=args.tiny)
    # Version 2 with a KL term keeps a frozen copy of the same snapshot. It is made before the run
    # directory exists, so a copy that does not fit leaves no directory behind.
    ref = frozen_reference(model) if (v2 and design["kl_beta"] > 0) else None
    load_seconds = time.time() - t_load
    out.mkdir(parents=True, exist_ok=True)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    eos_ids = eos_ids_of(tok)
    lr = design["lr"] if v2 else args.lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    sampler = Sampler()

    cfg = {**design, "lr": lr, "weight_decay": args.weight_decay, "clip_coef": args.clip_coef,
           "grad_clip": args.grad_clip, "eval_seed": args.eval_seed,
           "eval_chunk_sequences": args.eval_chunk_sequences, "eval_limit": args.eval_limit,
           "sampler": vars(sampler), "tiny": args.tiny, "device": dev, "eos_ids": eos_ids,
           "n_hard": len(hard), "n_random": len(rand), "short_threshold": short_threshold}
    if v2:
        cfg.update({"time_limit_minutes": args.time_limit_minutes,
                    "early_finish_margin_seconds": args.early_finish_margin_seconds,
                    "reference_loaded": ref is not None})
    (out / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    import transformers
    prov = {
        "reasoning_sources_digest": RT.reasoning_sources_digest(),
        "training_sources_digest": TSG.training_sources_digest(),
        "dataset_sha256": dataset_sha, "selection_sha256": RT.file_sha256(args.selection),
        "manifest_sha256": RT.file_sha256(args.manifest) if args.manifest else None,
        "model_id": design["model_id"], "model_revision": design["model_revision"],
        "model_snapshot": model_info["snapshot"], "model_file_sha256": model_info["file_sha256"],
        "git_sha": TSG.git_sha(), "python": sys.version.split()[0], "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__, "device": dev,
        "device_name": TSG.device_name(dev), "hostname": socket.gethostname(), "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps_completed": 0,
    }
    if v2:
        prov.update({"design_version": RT.DESIGN_VERSION_V2, "frozen_v2": frozen_v2,
                     "stopped_early": False, "early_finish": None})
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    timing = {"kind": design["kind"], "size": design["size"], "arm": design["arm"], "L": design["L"],
              "micro_batch": design["micro_batch"], "steps": design["steps"],
              "selection_sha256": prov["selection_sha256"], "device_name": prov["device_name"],
              "model_load_seconds": load_seconds, "step_seconds": [], "generation_seconds": [],
              "update_seconds": [], "eval_seconds": [], "peak_memory_reserved_bytes": None,
              "peak_memory_allocated_bytes": None, "steps_completed": 0}

    metrics = (out / "metrics.jsonl").open("w")
    completions = (out / "completions.jsonl").open("w")

    def peak_memory():
        if dev == "cuda":
            timing["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            timing["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())

    def write_timing():
        peak_memory()
        (out / "timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")

    def run_evals(step, final=False):
        # The final evaluations are the plan of the last step. An early finish runs them at the step it
        # stopped after, so a stopped run still ends with the hard set at 4 samples and the random set at 2.
        plan = RT.eval_plan(design["steps"] if final else step, design["steps"], design["mid_eval_every"])
        sets = [(name, hard if name == "hard" else rand, k) for name, k in plan]
        if not sets:
            return
        for rec in evaluate(model, tok, dev, sets, cands, eos_ids, design["L"], sampler, args.eval_seed,
                            args.eval_chunk_sequences, short_threshold, step, completions, timing):
            metrics.write(json.dumps(rec) + "\n")
        metrics.flush()
        completions.flush()

    model.train()
    run_evals(0)
    write_timing()
    # The full-evaluation time the early-finish guard reads is the measured step-0 evaluation, which
    # is the same plan the run must still pay at its end.
    full_eval_seconds = sum(e["seconds"] for e in timing["eval_seconds"] if e["step"] == 0)
    stopped_early = False
    K = RT.GROUP_SIZE
    for step in range(1, design["steps"] + 1):
        if v2 and args.time_limit_minutes is not None:
            mid = bool(RT.eval_plan(step, design["steps"], design["mid_eval_every"])) and step != design["steps"]
            need = early_finish_needed_seconds(timing["step_seconds"], full_eval_seconds, mid)
            remaining = args.time_limit_minutes * 60.0 - (time.time() - t_process) - args.early_finish_margin_seconds
            if need is not None and remaining < need:
                stopped_early = True
                prov["early_finish"] = {"before_step": step, "remaining_seconds": remaining, "needed_seconds": need,
                                        "mean_step_seconds": float(np.mean(timing["step_seconds"])),
                                        "full_eval_seconds": full_eval_seconds,
                                        "margin_seconds": args.early_finish_margin_seconds}
                print(json.dumps({"event": "early_finish", **prov["early_finish"]}), flush=True)
                break
        t0 = time.time()
        pids = schedule[step - 1]
        if len(pids) != RT.GROUPS_PER_STEP:
            raise RuntimeError("a schedule step does not hold six problems")
        model.eval()
        gen = batched_generate(model, tok, [encode_prompt(tok, cands[p]["prompt"]) for p in pids], K,
                               sampler, design["L"], dev)
        t_gen = time.time()
        groups_log, groups = [], []
        for pid, (seqs, plen) in zip(pids, gen):
            recs, mask = completion_records(tok, seqs, plen, eos_ids)
            score_texts(recs, RT.parse_fraction_str(cands[pid]["reference"]))
            fails = [r["F"] for r in recs]
            n_tok = [r["n_tokens"] for r in recs]
            r = RT.reasoning_rewards(fails, n_tok, design["form"], design["w"], design["w_task"], design["L"])
            a = group_advantages(r, fails, design)
            kept = RW.keep_group(design["group_filter"], r, [f == 0 for f in fails])
            g_log = {"pid": pid, "n_tokens": n_tok, "F": fails,
                     "finished": [x["finished"] for x in recs],
                     "strict": [x["strict"] for x in recs], "lenient": [x["lenient"] for x in recs],
                     "rewards": r.tolist(), "adv": a.tolist(), "kept": kept,
                     "all_fail": all(f == 1 for f in fails), "all_pass": all(f == 0 for f in fails)}
            if v2:
                zero_adv = RW.group_adv_is_zero(a)
                for_update = kept and not (design["skip_zero_adv"] and zero_adv)
                g_log.update({"zero_adv": zero_adv, "kept_for_update": for_update})
                groups.append(Group(pid, seqs, mask, torch.tensor(a, dtype=torch.float32, device=dev), kept,
                                    zero_adv=zero_adv, kept_for_update=for_update))
            else:
                groups.append(Group(pid, seqs, mask, torch.tensor(a, dtype=torch.float32, device=dev), kept))
            groups_log.append(g_log)
            for j, x in enumerate(recs):
                completions.write(json.dumps({"step": step, "phase": "train", "pid": pid, "sample": j,
                                              "token_ids": x["token_ids"], "text": x["text"]}) + "\n")
        model.train()
        if v2:
            upd = micro_batched_update_v2(model, ref, opt, groups, design["micro_batch"], args.clip_coef,
                                          args.grad_clip, design["kl_beta"])
        else:
            upd = micro_batched_update(model, opt, groups, design["micro_batch"], args.clip_coef, args.grad_clip)
        t_end = time.time()
        flat_f = [f for g in groups_log for f in g["F"]]
        flat_n = [n for g in groups_log for n in g["n_tokens"]]
        flat_a = np.array([x for g in groups_log for x in g["adv"]])
        fail = np.array(flat_f) == 1
        rec = {
            "step": step, "phase": "train",
            "accuracy": 1.0 - float(np.mean(flat_f)),
            "strict_parse_fail_rate": sum(1 for g in groups_log for s in g["strict"] if s is None) / len(flat_f),
            "truncation_rate": sum(1 for g in groups_log for x in g["finished"] if not x) / len(flat_f),
            "completion_tokens_mean": float(np.mean(flat_n)),
            "n_all_fail_groups": sum(1 for g in groups_log if g["all_fail"]),
            "n_mixed_groups": sum(1 for g in groups_log if not g["all_fail"] and not g["all_pass"]),
            "n_all_pass_groups": sum(1 for g in groups_log if g["all_pass"]),
            "all_fail_completion_tokens_mean": (float(np.mean([n for g in groups_log if g["all_fail"]
                                                               for n in g["n_tokens"]]))
                                                if any(g["all_fail"] for g in groups_log) else None),
            "frac_fail_pos_adv": float((flat_a[fail] > 0).mean()) if fail.any() else None,
            "reward_mean": float(np.mean([x for g in groups_log for x in g["rewards"]])),
            "opt_step_taken": upd["opt_step_taken"], "n_kept_groups": upd["n_kept_groups"],
            "grad_norm_preclip": upd["grad_norm_preclip"], "pg_loss": upd["pg_loss"],
            "groups": groups_log,
        }
        if v2:
            rec.update({k: upd[k] for k in ("n_zero_adv_groups", "n_update_groups", "loss", "kl_loss",
                                            "kl_k3_tok_mean", "kl_seq_mean", "entropy_tok_mean")})
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()
        completions.flush()
        timing["step_seconds"].append(t_end - t0)
        timing["generation_seconds"].append(t_gen - t0)
        timing["update_seconds"].append(t_end - t_gen)
        timing["steps_completed"] = step
        run_evals(step)
        write_timing()
        prov["steps_completed"] = step
        if step == 1 or step % 10 == 0:
            print(json.dumps({"step": step, "seconds": round(t_end - t0, 3),
                              "peak_reserved": timing["peak_memory_reserved_bytes"]}), flush=True)

    if stopped_early:
        # The run ends as a complete run ends: with the final evaluations, at the last completed step.
        run_evals(prov["steps_completed"], final=True)
        prov["stopped_early"] = True
    metrics.close()
    completions.close()
    write_timing()
    prov["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    prov["final_param_checksum"] = TSG.param_checksum(model)
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")
    if prov["steps_completed"] != design["steps"] and not stopped_early:
        raise RuntimeError("the run did not complete its step count")
    (out / "DONE").write_text("ok\n")
    print(json.dumps({"event": "done", "out": str(out), "steps": design["steps"],
                      "steps_completed": prov["steps_completed"], "stopped_early": stopped_early}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
