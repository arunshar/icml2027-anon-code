"""The remedy flags in the trainer, on the random tiny model.

Four kinds of check:
  * every variant and filter runs, and what it logs is what the reward code recomputes;
  * a filter that drops every group leaves the weights untouched, and a partial filter divides the
    loss by the completions it kept;
  * a zero-advantage step still moves the weights through AdamW's momentum, and a skipped step
    does not, which is the distinction between "the gradient is zero" and "the update is zero";
  * the KL penalty is exactly zero, with an exactly zero gradient, at the reference, and equals an
    explicit construction once the policy has moved.
"""
import json
import os
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

torch = pytest.importorskip("torch")

import analyze  # noqa: E402
import rewards as RW  # noqa: E402
import tasks  # noqa: E402
import train_speed_grpo as T  # noqa: E402
from speed_task import PROMPTS  # noqa: E402

BASE = ["--arm", "binary", "--w", "5", "--vmax", "5", "--tiny", "--device", "cpu",
        "--eval-every", "0", "--batch-prompts", "2", "--seed", "0"]


def _read(out):
    recs = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    cfg = json.load(open(os.path.join(out, "config.json")))
    prov = json.load(open(os.path.join(out, "provenance.json")))
    return (cfg, prov, [r for r in recs if r["phase"] == "train"],
            [r for r in recs if r["phase"] == "eval"])


def _run(tmp_path, name, extra, steps=2):
    out = tmp_path / name
    assert T.main(BASE + ["--steps", str(steps), "--out-dir", str(out)] + list(extra)) == 0
    return _read(out)


def _check_logged_against_recomputation(cfg, train):
    tspec = tasks.get(cfg["task"])
    spec = None if tspec.name == "speed" else tspec
    for rec in train:
        for gi, g in enumerate(rec["groups"]):
            items = g[tspec.ITEMS_KEY]
            if spec is not None:
                items = [None if x is None else tuple(x) for x in items]
            assert g["prompt_index"] == gi
            r = RW.group_rewards(items, cfg["arm"], cfg["w"], cfg["vmax"], spec=spec,
                                 w_task=cfg["w_task"])
            assert r.tolist() == g["rewards"]
            anchor = RW.reward_anchor(cfg["adv_variant"], cfg["arm"], cfg["w"], cfg["vmax"],
                                      spec=spec, w_task=cfg["w_task"],
                                      instance=tspec.instance_for(gi))
            assert g["anchor"] == anchor
            assert (anchor is None) == (cfg["adv_variant"] not in RW.ANCHORED_VARIANTS)
            assert RW.advantage_variant(r, cfg["adv_variant"], anchor).tolist() == g["adv"]
            assert g["kept"] == RW.keep_group(cfg["group_filter"], r,
                                              RW.group_verdicts(items, cfg["vmax"], spec=spec))
            assert len(g["n_tokens"]) == cfg["group_size"]
            assert all(1 <= n <= cfg["max_new_tokens"] for n in g["n_tokens"])
        groups = rec["groups"]
        n_kept = sum(g["kept"] for g in groups)
        assert rec["n_kept_groups"] == n_kept
        assert rec["opt_step_taken"] is (n_kept > 0)
        assert (rec["grad_norm_preclip"] is None) is (n_kept == 0)
        assert rec["n_mixed_groups"] == sum(g["has_violator"] and not g["all_violating"]
                                            for g in groups)
        assert rec["n_all_feasible_groups"] == sum(not g["has_violator"] for g in groups)
        assert rec["completion_tokens_mean"] == pytest.approx(
            np.mean([n for g in groups for n in g["n_tokens"]]))
        assert rec["kl_k3_tok_mean"] is None and rec["kl_loss"] is None


CASES = [("grpo", "none"), ("drgrpo", "none"), ("ngrpo_feasible", "none"), ("ngrpo_sup", "none"),
         ("grpo", "verdict_mixed"), ("grpo", "reward_spread")]


@pytest.mark.parametrize("variant,group_filter", CASES)
def test_every_variant_and_filter_logs_what_the_reward_code_recomputes(tmp_path, variant,
                                                                         group_filter):
    cfg, prov, train, _ = _run(tmp_path, f"{variant}_{group_filter}",
                               ["--adv-variant", variant, "--group-filter", group_filter])
    assert cfg["adv_variant"] == variant and cfg["group_filter"] == group_filter
    assert cfg["kl_beta"] == 0.0
    assert [r["step"] for r in train] == [1, 2]
    _check_logged_against_recomputation(cfg, train)
    assert prov["training_sources_digest"] == T.training_sources_digest()


def test_the_capacity_anchor_follows_the_prompt_instance(tmp_path):
    out = tmp_path / "cap"
    # The no-penalty arm's largest reward is the whole instance's value over its optimum, which
    # differs between instances. The residual arm's is the feasible optimum, 1.0 everywhere.
    argv = ["--task", "capacity", "--arm", "none", "--w", "0", "--vmax", "0.03", "--tiny",
            "--device", "cpu", "--eval-every", "0", "--batch-prompts", "3", "--seed", "0",
            "--steps", "1", "--max-new-tokens", "8", "--backward-per-group",
            "--adv-variant", "ngrpo_sup", "--group-filter", "verdict_mixed", "--out-dir", str(out)]
    assert T.main(argv) == 0
    cfg, _, train, _ = _read(out)
    _check_logged_against_recomputation(cfg, train)
    anchors = [g["anchor"] for g in train[0]["groups"]]
    assert len(set(anchors)) > 1, "the capacity instances have different sup rewards"


def test_the_default_path_takes_every_group_and_every_step(tmp_path):
    cfg, _, train, _ = _run(tmp_path, "plain", [])
    assert (cfg["adv_variant"], cfg["group_filter"], cfg["kl_beta"]) == ("grpo", "none", 0.0)
    for rec in train:
        assert rec["opt_step_taken"] is True
        assert rec["n_kept_groups"] == len(rec["groups"])
        assert all(g["kept"] is True and g["anchor"] is None for g in rec["groups"])
    _check_logged_against_recomputation(cfg, train)


def _initial_checksum(argv):
    args = T.build_arg_parser().parse_args(argv)
    T.seed_everything(args.seed)
    _, model, _ = T.build_models(args, "cpu")
    return T.param_checksum(model)


@pytest.mark.parametrize("per_group", [False, True])
def test_a_run_whose_every_group_is_filtered_never_moves_a_weight(tmp_path, per_group):
    """A8: a filter that keeps nothing must leave the final evaluation equal to the step-0 one."""
    out = tmp_path / f"filtered_{per_group}"
    argv = BASE + ["--steps", "2", "--group-filter", "reward_spread", "--eval-every", "2",
                   "--eval-samples", "2", "--out-dir", str(out)]
    if per_group:
        argv.append("--backward-per-group")
    assert T.main(argv) == 0
    _, prov, train, evals = _read(out)
    groups = [g for r in train for g in r["groups"]]
    assert groups and not any(g["kept"] for g in groups), \
        "precondition: the tiny model at this seed emits no reward spread, so every group drops"
    assert all(r["opt_step_taken"] is False and r["pg_loss"] is None for r in train)
    assert prov["final_param_checksum"] == _initial_checksum(argv)
    assert [e["step"] for e in evals] == [0, 2]
    assert evals[0]["eval_speeds"] == evals[1]["eval_speeds"]


# ---------------------------------------------------------------- the update itself


class _CaptureOpt:
    """Stands in for AdamW when a test needs the gradient the optimizer would have received."""

    def __init__(self, model):
        self.model, self.grads, self.steps = model, None, 0

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def step(self):
        self.grads = [None if p.grad is None else p.grad.detach().clone()
                      for p in self.model.parameters()]
        self.steps += 1


def _tiny(with_ref=False):
    argv = BASE + ["--out-dir", os.devnull] + ([] if with_ref else ["--no-ref"])
    args = T.build_arg_parser().parse_args(argv)
    T.seed_everything(args.seed)
    tok, model, ref = T.build_models(args, "cpu")
    return args, tok, model, ref


def _batch(tok, model, args, n_groups=3, max_new_tokens=8):
    args.max_new_tokens = max_new_tokens
    eos = sorted({i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None})
    seqs, masks = [], []
    for pi in range(n_groups):
        ids = T.chat_ids(tok, PROMPTS[pi], "cpu")
        gen = T.sample_completions(model, tok, ids, args, args.group_size)
        seqs.append(gen)
        masks.append(T.completion_mask(gen, ids.shape[1], eos))
    return seqs, masks


def _explicit_pg_grads(model, seqs, masks, advs, kept, clip, denominator):
    model.zero_grad(set_to_none=True)
    terms = []
    for s, m, a, k in zip(seqs, masks, advs, kept):
        if not k:
            continue
        with torch.no_grad():
            old = T.seq_logprobs(model, s, m)
        new = T.seq_logprobs(model, s, m)
        ratio = (new - old).exp()
        terms.append(-torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a))
    (torch.cat(terms).sum() / denominator).backward()
    return [p.grad.detach().clone() for p in model.parameters()]


@pytest.mark.parametrize("per_group", [False, True])
def test_a_partial_filter_divides_by_the_kept_completions(per_group):
    args, tok, model, _ = _tiny()
    args.grad_clip = 1e30                      # the clip multiplies by exactly 1.0
    args.backward_per_group = per_group
    seqs, masks = _batch(tok, model, args)
    g = torch.Generator().manual_seed(11)
    advs = [torch.randn(args.group_size, generator=g) for _ in seqs]
    kept = [True, False, True]
    n_kept = sum(args.group_size for k in kept if k)
    n_all = args.group_size * len(kept)

    want = _explicit_pg_grads(model, seqs, masks, advs, kept, args.clip_coef, n_kept)
    opt = _CaptureOpt(model)
    st = T.policy_update(model, None, opt, seqs, masks, advs, kept, args)
    assert opt.steps == 1 and st["opt_step_taken"] and st["n_kept_groups"] == 2
    for a, b in zip(opt.grads, want):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-8)
    # the denominator is the kept count, not every completion in the step
    over_all = _explicit_pg_grads(model, seqs, masks, advs, kept, args.clip_coef, n_all)
    assert any(not torch.allclose(a, b, rtol=1e-3, atol=1e-8) for a, b in zip(opt.grads, over_all))
    # and a dropped group changes the gradient
    every = _explicit_pg_grads(model, seqs, masks, advs, [True] * 3, args.clip_coef, n_all)
    assert any(not torch.allclose(a, b, rtol=1e-3, atol=1e-8) for a, b in zip(opt.grads, every))


def test_the_single_backward_path_is_the_explicit_mean_exactly():
    args, tok, model, _ = _tiny()
    args.grad_clip = 1e30
    seqs, masks = _batch(tok, model, args)
    g = torch.Generator().manual_seed(12)
    advs = [torch.randn(args.group_size, generator=g) for _ in seqs]
    kept = [True] * 3
    model.zero_grad(set_to_none=True)
    terms = []
    for s, m, a in zip(seqs, masks, advs):
        with torch.no_grad():
            old = T.seq_logprobs(model, s, m)
        new = T.seq_logprobs(model, s, m)
        ratio = (new - old).exp()
        terms.append(-torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip_coef,
                                                       1 + args.clip_coef) * a))
    torch.cat(terms).mean().backward()
    want = [p.grad.detach().clone() for p in model.parameters()]
    opt = _CaptureOpt(model)
    T.policy_update(model, None, opt, seqs, masks, advs, kept, args)
    assert all(torch.equal(a, b) for a, b in zip(opt.grads, want))


def test_momentum_moves_the_weights_on_a_zero_gradient_step_and_a_skipped_step_does_not():
    """Z1. The gradient of a zero-advantage step is zero, and AdamW still moves the weights."""
    def two_steps(second_kept):
        args, tok, model, _ = _tiny()
        seqs, masks = _batch(tok, model, args)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        push = [torch.tensor([1.5, -0.5, -0.5, -0.5]) for _ in seqs]
        T.policy_update(model, None, opt, seqs, masks, push, [True] * len(seqs), args)
        after_first = T.param_checksum(model)
        zero = [torch.zeros(args.group_size) for _ in seqs]
        st = T.policy_update(model, None, opt, seqs, masks, zero, second_kept, args)
        steps = {int(s["step"]) for s in opt.state.values()}
        return after_first, T.param_checksum(model), st, steps

    first, second, st, steps = two_steps([True, True, True])
    assert st["opt_step_taken"] is True and st["grad_norm_preclip"] == 0.0
    assert second != first, "a zero-gradient AdamW step after a nonzero one must move the weights"
    assert steps == {2}
    first_b, second_b, st_b, steps_b = two_steps([False, False, False])
    assert first_b == first, "the two runs share the first step"
    assert st_b["opt_step_taken"] is False and st_b["grad_norm_preclip"] is None
    assert second_b == first_b, "a skipped step must leave the weights exactly where they were"
    assert steps_b == {1}, "a skipped step must not advance the optimizer's step count"


# ---------------------------------------------------------------- KL penalty


@pytest.mark.parametrize("per_group", [False, True])
def test_the_kl_penalty_and_its_gradient_are_exactly_zero_at_the_reference(per_group):
    args, tok, model, ref = _tiny(with_ref=True)
    args.kl_beta, args.grad_clip, args.backward_per_group = 0.04, 1e30, per_group
    seqs, masks = _batch(tok, model, args)
    zero = [torch.zeros(args.group_size) for _ in seqs]
    opt = _CaptureOpt(model)
    st = T.policy_update(model, ref, opt, seqs, masks, zero, [True] * len(seqs), args)
    assert st["kl_loss"] == 0.0 and st["kl_k3_tok_mean"] == 0.0
    assert all(torch.count_nonzero(g) == 0 for g in opt.grads if g is not None)


def _explicit_kl(model, ref, seqs, masks, beta):
    vals, k3_tok_sum = [], 0.0
    for s, m in zip(seqs, masks):
        tgt = s[:, 1:].unsqueeze(-1)
        new = torch.log_softmax(model(s).logits[:, :-1, :].float(), dim=-1).gather(-1, tgt).squeeze(-1)
        with torch.no_grad():
            old = torch.log_softmax(ref(s).logits[:, :-1, :].float(), dim=-1).gather(-1, tgt).squeeze(-1)
        x = old - new
        k3 = torch.where(m, torch.exp(x) - x - 1.0, torch.zeros_like(x))
        vals.append(k3.sum(dim=1))
        k3_tok_sum += float(k3.detach().sum())
    n_tok = sum(int(m.sum()) for m in masks)
    return beta * torch.cat(vals).mean(), k3_tok_sum / n_tok


@pytest.mark.parametrize("per_group", [False, True])
def test_the_kl_penalty_equals_an_explicit_k3_construction_after_the_policy_moves(per_group):
    args, tok, model, ref = _tiny(with_ref=True)
    args.kl_beta, args.grad_clip, args.backward_per_group = 0.04, 1e30, per_group
    seqs, masks = _batch(tok, model, args)
    with torch.no_grad():
        g = torch.Generator().manual_seed(7)
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    model.zero_grad(set_to_none=True)
    value, k3_tok = _explicit_kl(model, ref, seqs, masks, args.kl_beta)
    value.backward()
    want = [p.grad.detach().clone() for p in model.parameters()]
    value = float(value.detach())
    assert value > 0

    zero = [torch.zeros(args.group_size) for _ in seqs]
    opt = _CaptureOpt(model)
    st = T.policy_update(model, ref, opt, seqs, masks, zero, [True] * len(seqs), args)
    assert st["kl_loss"] == pytest.approx(value, rel=1e-6)
    assert st["kl_k3_tok_mean"] == pytest.approx(k3_tok, rel=1e-6)
    # The entries are about 1e-7 in float32, where an element-wise relative test measures rounding.
    # The two constructions mask differently (a product against a where), so compare by norm.
    total = torch.sqrt(sum((b.double() ** 2).sum() for b in want))
    diff = torch.sqrt(sum(((a.double() - b.double()) ** 2).sum() for a, b in zip(opt.grads, want)))
    assert float(total) > 0
    assert float(diff) <= 1e-4 * float(total), f"gradient differs by {float(diff / total):.2e} of its norm"


def test_a_kl_run_logs_zero_penalty_at_step_one(tmp_path):
    _, _, train, _ = _run(tmp_path, "kl", ["--kl-beta", "0.04"])
    assert train[0]["kl_loss"] == 0.0 and train[0]["kl_k3_tok_mean"] == 0.0
    assert all(r["kl_loss"] is not None and r["kl_loss"] >= 0.0 for r in train)


def test_the_beta_zero_path_never_calls_the_token_logprob_function(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("token_logprobs was called on the beta = 0 path")
    monkeypatch.setattr(T, "token_logprobs", boom)
    _run(tmp_path, "plain", [], steps=1)
    _run(tmp_path, "filtered", ["--group-filter", "verdict_mixed"], steps=1)


@pytest.mark.parametrize("extra", [
    ["--kl-beta", "0.04", "--no-ref"],
    ["--kl-beta", "0.04", "--adv-variant", "ngrpo_feasible"],
    ["--kl-beta", "0.04", "--group-filter", "verdict_mixed"],
    ["--kl-beta", "-0.01"],
])
def test_undefined_flag_combinations_are_refused_before_anything_is_written(tmp_path, extra):
    out = tmp_path / "refused"
    with pytest.raises(SystemExit) as e:
        T.main(BASE + ["--steps", "1", "--out-dir", str(out)] + extra)
    assert e.value.code == 2
    assert not out.exists()


# ---------------------------------------------------------------- determinism and provenance


@pytest.mark.parametrize("extra", [["--adv-variant", "ngrpo_feasible"],
                                   ["--group-filter", "verdict_mixed"]])
def test_same_seed_same_run_under_a_remedy_flag(tmp_path, extra):
    def strip(train):
        return [{k: v for k, v in r.items() if k != "s_per_step"} for r in train]
    _, pa, ta, _ = _run(tmp_path, "a", extra)
    _, pb, tb, _ = _run(tmp_path, "b", extra)
    assert strip(ta) == strip(tb)
    assert pa["final_param_checksum"] == pb["final_param_checksum"]


def test_the_trainer_and_the_analysis_declare_the_same_training_sources():
    assert tuple(T.TRAINING_SOURCES) == tuple(analyze.TRAINING_SOURCES)


def test_the_digest_follows_content_and_survives_a_missing_file(tmp_path):
    for rel in T.TRAINING_SOURCES:
        src = os.path.join(HERE, rel[len("experiments/cancellation/"):])
        dst = tmp_path / rel[len("experiments/cancellation/"):]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    here = T.training_sources_digest()
    assert T.training_sources_digest(tmp_path) == here
    target = tmp_path / "rewards.py"
    target.write_text(target.read_text() + "\n")
    changed = T.training_sources_digest(tmp_path)
    assert changed != here
    target.unlink()
    assert T.training_sources_digest(tmp_path) not in (here, changed)
