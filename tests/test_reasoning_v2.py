"""Design version 2 of the GSM8K block (ICML 2027, November 2026), on the random tiny model.

Seven things are checked, and every one of them leaves the version-1 path alone:
  * the k3 KL penalty is exactly zero with a zero gradient at the reference, and positive once the
    policy has moved, where it equals an explicit construction (the speed trainer's test, mirrored);
  * the restored arm logs the advantages that rewards.advantage_fixedref recomputes from the logged
    verdicts and rewards, and differs from plain only on homogeneous-verdict groups;
  * the zero-advantage skip behaves as in the speed trainer: a skipped step leaves the parameter
    checksum where it was, and the same step without the skip moves it through AdamW's momentum;
  * the per-token entropy is logged, finite, and equals a recomputation at step one;
  * a version-1 manifest row still validates, and a version-1 tiny run writes the metrics.jsonl bytes
    it wrote before version 2 existed (golden hashes recorded on 2026-11-06, below);
  * a version-2 row refuses to run without configs/reasoning_frozen_v2.json or with a stale one;
  * the early-finish guard stops a run that cannot fit another step plus the final evaluations, runs
    those evaluations, records stopped_early, and the analysis shows the run.

The tokenizer comes from the pinned 0.5B snapshot in the local cache. These tests skip when it is absent.
"""
import copy
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import analyze_reasoning as A  # noqa: E402
import reasoning_task as RT  # noqa: E402
import rewards as RW  # noqa: E402
import train_reasoning_grpo as TR  # noqa: E402
import train_speed_grpo as TSG  # noqa: E402
from reasoning_fixtures import write_tiny_design  # noqa: E402

HERE = Path(__file__).resolve().parent.parent


def _has_tokenizer():
    try:
        TR.model_snapshot(RT.MODELS["0.5b"]["model_id"], RT.MODELS["0.5b"]["revision"])
        return True
    except TR.Refused:
        return False


pytestmark = pytest.mark.skipif(not _has_tokenizer(), reason="the pinned 0.5B tokenizer is not cached here")

# The SHA-256 of metrics.jsonl and completions.jsonl written by a version-1 tiny run of the trainer at
# the commit before the version-2 code was added (2026-11-06, worktree icml2027, after rewards.py and
# train_speed_grpo.py had already moved). The run is write_tiny_design plus the flags in GOLDEN_ARGV.
# The "plain" case is the arm as it runs. The tiny model never emits an end token, so every group has
# identical lengths and a zero advantage, and the pinned update is the zero-gradient AdamW step. The
# "mixed_plain" case sets the verdict from the parity of the token ids, as test_reasoning_trainer.py
# does, so every step takes a real update. Floating-point results of a CPU forward pass depend on the
# library versions and on the machine, so the comparison skips, and says why, when either differs.
GOLDEN_VERSIONS = {"torch": "2.14.0", "transformers": "5.17.0", "numpy": "2.5.3", "python": "3.12.13",
                   "machine": "arm64"}
GOLDEN_ARGV = {"plain": ["--steps", "3", "--L", "16", "--seed", "0", "--arm", "plain", "--mid-eval-every", "1"],
               "mixed_plain": ["--steps", "3", "--L", "16", "--seed", "0", "--arm", "plain", "--mid-eval-every", "0"]}
GOLDEN_SHA256 = {
    "plain": {"metrics.jsonl": "c8061158b3d17f453b98f63745e201aee20d90388e890257e6301c9b5807e0e6",
              "completions.jsonl": "d457da499494750d358fa51f9191f452f38774f1150cc8ee892df39230d52966",
              "final_param_checksum": "45fa6318d7ad72bea8e16f7e2cb4d9a6c1d8f12ff0265fdcfc511d2897babe18"},
    "mixed_plain": {"metrics.jsonl": "c47bb110a19713ec9748c957d08fc026c66bac132d5e0d0cb54b478c569f1624",
                    "completions.jsonl": "d3a71549d074b367e29037826b957b2735d4649a7c51ec59340bacfed798016b",
                    "final_param_checksum": "0fe75711304031665936545778e5cda570d04532c0e4f20ae39cd3e20e0d0f6c"},
}


def _versions():
    import numpy
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": numpy.__version__,
            "python": platform.python_version(), "machine": platform.machine()}


# ------------------------------------------------------------------ helpers


def write_frozen_v2(path: Path, training=None, reasoning=None) -> Path:
    doc = {"schema": TR.FROZEN_V2_SCHEMA, "design_version": 2,
           "training_sources_digest": training or TSG.training_sources_digest(),
           "reasoning_sources_digest": reasoning or RT.reasoning_sources_digest()}
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return path


def base_argv(tmp: Path, out: Path, steps=3, arm="plain", version=2, extra=()):
    if not (tmp / "ds.json").exists():
        write_tiny_design(tmp)
    argv = ["--dataset", str(tmp / "ds.json"), "--selection", str(tmp / "sel.json"), "--out-dir", str(out),
            "--tiny", "--device", "cpu", "--steps", str(steps), "--L", "16", "--seed", "0", "--arm", arm,
            "--mid-eval-every", "0"]
    if version == 2:
        argv += ["--design-version", "2", "--frozen-v2", str(write_frozen_v2(tmp / "frozen_v2.json"))]
    return argv + list(extra)


def records(out: Path, phase: str):
    return [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines() if json.loads(x)["phase"] == phase]


def prov(out: Path) -> dict:
    return json.loads((out / "provenance.json").read_text())


def cfg(out: Path) -> dict:
    return json.loads((out / "config.json").read_text())


def _parity_verdicts(records_, reference):
    """Score the real parses, then set F from the parity of the token ids so that groups are mixed."""
    _real_score(records_, reference)
    for r in records_:
        r["F"] = sum(r["token_ids"]) % 2


_real_score = TR.score_texts


def _tiny_model():
    TSG.seed_everything(0)
    tok, model, _ = TR.load_model(RT.MODELS["0.5b"]["model_id"], RT.MODELS["0.5b"]["revision"], "cpu", tiny=True)
    return tok, model


def _groups(tok, model, n_prompts=2, max_new=12, adv=None, kept_for_update=True):
    prompts = [TR.encode_prompt(tok, RT.prompt_text(q)) for q in
               ("What is 7 minus 2?", "How many legs do 3 cats have?", "What is 4 times 5?")[:n_prompts]]
    gen = TR.batched_generate(model, tok, prompts, RT.GROUP_SIZE, TR.Sampler(), max_new, "cpu")
    eos = TR.eos_ids_of(tok)
    groups = []
    for i, (seqs, plen) in enumerate(gen):
        mask = TSG.completion_mask(seqs, plen, eos)
        a = torch.zeros(RT.GROUP_SIZE) if adv is None else adv[i]
        groups.append(TR.Group(f"p{i}", seqs, mask, a, True, zero_adv=bool(RW.group_adv_is_zero(a.numpy())),
                               kept_for_update=kept_for_update))
    return groups


# ------------------------------------------------------------------ the KL penalty


def test_the_kl_penalty_and_its_gradient_are_exactly_zero_at_the_reference():
    tok, model = _tiny_model()
    model.train()
    ref = TR.frozen_reference(model)
    assert not any(p.requires_grad for p in ref.parameters()) and not ref.training
    groups = _groups(tok, model)
    model.zero_grad(set_to_none=True)
    st = TR.accumulate_gradients_v2(model, ref, groups, RT.MICRO_BATCH, 0.2, 0.02)
    assert st["kl_loss"] == 0.0 and st["kl_k3_tok_mean"] == 0.0 and st["kl_seq_mean"] == 0.0
    assert st["n_update_groups"] == 2 and st["opt_step_taken"] is True
    assert all(torch.count_nonzero(p.grad) == 0 for p in model.parameters() if p.grad is not None)
    assert np.isfinite(st["entropy_tok_mean"]) and st["entropy_tok_mean"] > 0


def _explicit_kl(model, ref, groups, beta):
    vals, k3_tok_sum, n_tok = [], 0.0, 0
    for g in groups:
        s, m = g.seqs, g.mask
        tgt = s[:, 1:].unsqueeze(-1)
        new = torch.log_softmax(model(s).logits[:, :-1, :].float(), dim=-1).gather(-1, tgt).squeeze(-1)
        with torch.no_grad():
            old = torch.log_softmax(ref(s).logits[:, :-1, :].float(), dim=-1).gather(-1, tgt).squeeze(-1)
        x = old - new
        k3 = torch.where(m, torch.exp(x) - x - 1.0, torch.zeros_like(x))
        vals.append(k3.sum(dim=1))
        k3_tok_sum += float(k3.detach().sum())
        n_tok += int(m.sum())
    return beta * torch.cat(vals).mean(), k3_tok_sum / n_tok


def test_the_kl_penalty_equals_an_explicit_k3_construction_after_the_policy_moves():
    tok, model = _tiny_model()
    model.train()
    ref = TR.frozen_reference(model)
    groups = _groups(tok, model)
    with torch.no_grad():
        g = torch.Generator().manual_seed(7)
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    model.zero_grad(set_to_none=True)
    value, k3_tok = _explicit_kl(model, ref, groups, 0.02)
    value.backward()
    want = [p.grad.detach().clone() for p in model.parameters()]
    value = float(value.detach())
    assert value > 0
    model.zero_grad(set_to_none=True)
    st = TR.accumulate_gradients_v2(model, ref, groups, RT.MICRO_BATCH, 0.2, 0.02)
    assert st["kl_loss"] == pytest.approx(value, rel=1e-6)
    assert st["kl_k3_tok_mean"] == pytest.approx(k3_tok, rel=1e-6)
    assert st["kl_seq_mean"] != 0.0
    # The entries are about 1e-7 in float32, so compare the gradients by norm, as the speed test does.
    got = [p.grad.detach().clone() for p in model.parameters()]
    total = torch.sqrt(sum((b.double() ** 2).sum() for b in want))
    diff = torch.sqrt(sum(((a.double() - b.double()) ** 2).sum() for a, b in zip(got, want)))
    assert float(total) > 0 and float(diff) <= 1e-4 * float(total)


def test_a_kl_run_loads_a_reference_logs_zero_at_step_one_and_a_positive_penalty_later(tmp_path):
    out = tmp_path / "kl"
    assert TR.main(base_argv(tmp_path, out, extra=["--kl-beta", "0.02"])) == 0
    c = cfg(out)
    assert c["design_version"] == 2 and c["kl_beta"] == 0.02 and c["reference_loaded"] is True
    train = records(out, "train")
    assert train[0]["kl_loss"] == 0.0 and train[0]["kl_k3_tok_mean"] == 0.0 and train[0]["kl_seq_mean"] == 0.0
    assert all(r["kl_loss"] is not None and r["kl_loss"] >= 0.0 for r in train)
    assert all(np.isfinite(r["entropy_tok_mean"]) for r in train)
    assert prov(out)["design_version"] == 2 and prov(out)["stopped_early"] is False
    # Without a KL term no reference is loaded and the KL fields are None, while the entropy is logged.
    out0 = tmp_path / "nokl"
    assert TR.main(base_argv(tmp_path, out0)) == 0
    assert cfg(out0)["reference_loaded"] is False
    for r in records(out0, "train"):
        assert r["kl_loss"] is None and r["kl_k3_tok_mean"] is None and r["kl_seq_mean"] is None
        assert np.isfinite(r["entropy_tok_mean"])


# ------------------------------------------------------------------ the restored arm


@pytest.fixture(scope="module")
def mixed_v2_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mixed_v2")
    write_tiny_design(tmp)
    mp = pytest.MonkeyPatch()
    mp.setattr(TR, "score_texts", _parity_verdicts)
    out = {}
    try:
        for arm in ("plain", "restored", "vfilter", "verdict_only"):
            d = tmp / arm
            assert TR.main(base_argv(tmp, d, steps=4, arm=arm)) == 0
            out[arm] = d
    finally:
        mp.undo()
    return out


def test_the_restored_arm_logs_what_advantage_fixedref_recomputes(mixed_v2_runs):
    d = mixed_v2_runs["restored"]
    c = cfg(d)
    assert c["adv_variant"] == "grpo_fixedref" and c["fixedref_c"] == 2.0 and c["arm"] == "restored"
    homogeneous = 0
    for rec in records(d, "train"):
        for g in rec["groups"]:
            r = RT.reasoning_rewards(g["F"], g["n_tokens"], c["form"], c["w"], c["w_task"], c["L"])
            assert r.tolist() == g["rewards"]
            verdicts = [f == 0 for f in g["F"]]
            want = RW.advantage_fixedref(r, verdicts, c["fixedref_c"])
            assert want.tolist() == g["adv"]
            shift = np.asarray(g["adv"]) - RW.advantage(r)
            if all(verdicts) or not any(verdicts):
                homogeneous += 1
                assert shift == pytest.approx((2.0 if all(verdicts) else -2.0), abs=1e-12)
                assert g["zero_adv"] is False
            else:
                assert np.array_equal(shift, np.zeros_like(shift))
            assert g["zero_adv"] is RW.group_adv_is_zero(g["adv"])
            assert g["kept"] is True and g["kept_for_update"] is True
        assert rec["n_update_groups"] == rec["n_kept_groups"] == len(rec["groups"])
        assert rec["n_zero_adv_groups"] == sum(g["zero_adv"] for g in rec["groups"])
    assert homogeneous > 0, "precondition: the parity verdicts give at least one homogeneous group"


def test_the_restored_arm_differs_from_plain_only_on_homogeneous_groups_at_step_one(mixed_v2_runs):
    plain = records(mixed_v2_runs["plain"], "train")[0]
    restored = records(mixed_v2_runs["restored"], "train")[0]
    for gp, gr in zip(plain["groups"], restored["groups"]):
        assert gp["n_tokens"] == gr["n_tokens"] and gp["F"] == gr["F"] and gp["rewards"] == gr["rewards"]
        v = [f == 0 for f in gr["F"]]
        if all(v) or not any(v):
            assert np.asarray(gr["adv"]) - np.asarray(gp["adv"]) == pytest.approx((2.0 if all(v) else -2.0), abs=1e-12)
        else:
            assert gr["adv"] == gp["adv"]


def test_every_version_2_arm_logs_its_own_rule_and_the_filter_decision(mixed_v2_runs):
    for arm, d in mixed_v2_runs.items():
        c = cfg(d)
        assert c["design_version"] == 2 and c["skip_zero_adv"] is False
        for rec in records(d, "train"):
            for g in rec["groups"]:
                assert g["kept"] == RW.keep_group(RT.ARMS[arm]["group_filter"], g["rewards"], [f == 0 for f in g["F"]])
                assert g["kept_for_update"] is g["kept"], "with the skip off the filter decision is the update decision"
            assert rec["n_update_groups"] == rec["n_kept_groups"] == sum(g["kept"] for g in rec["groups"])
            assert rec["opt_step_taken"] is (rec["n_update_groups"] > 0)


# ------------------------------------------------------------------ the zero-advantage skip

PUSH = np.array([1.5, -0.5, -0.5, -0.5])


def _force_advantages(monkeypatch, n_pushed_calls):
    """Replace the advantage with a pushed vector for the first calls and zeros afterwards.

    The trainer calls group_advantages once per group per step, six groups per step, so the first six
    calls are step one. This is the speed trainer's technique in test_fixedref_and_skip.py."""
    calls = {"n": 0}

    def fake(r, fails, design):
        calls["n"] += 1
        return PUSH.copy() if calls["n"] <= n_pushed_calls else np.zeros(len(r))

    monkeypatch.setattr(TR, "group_advantages", fake)


def test_skip_zero_adv_takes_no_step_and_leaves_the_checksum_where_it_was(tmp_path, monkeypatch):
    """Step one has pushed advantages in every run, so AdamW's first moment is nonzero afterwards. Step
    two has zero advantages in every group. With the skip on, no group enters the update and the
    weights stay at their step-one values. Without the skip the zero-gradient step still moves the
    weights through the momentum, which the WO-I collapse investigation named as a drift driver."""
    G = RT.GROUPS_PER_STEP
    _force_advantages(monkeypatch, G)
    one = tmp_path / "one"
    assert TR.main(base_argv(tmp_path, one, steps=1)) == 0
    after_first = prov(one)["final_param_checksum"]
    assert records(one, "train")[0]["grad_norm_preclip"] > 0

    _force_advantages(monkeypatch, G)
    skip = tmp_path / "skip"
    assert TR.main(base_argv(tmp_path, skip, steps=2, extra=["--skip-zero-adv"])) == 0
    assert cfg(skip)["skip_zero_adv"] is True
    second = records(skip, "train")[1]
    assert second["opt_step_taken"] is False and second["grad_norm_preclip"] is None and second["pg_loss"] is None
    assert second["n_kept_groups"] == G and second["n_zero_adv_groups"] == G and second["n_update_groups"] == 0
    for g in second["groups"]:
        assert g["adv"] == [0.0] * 4 and g["zero_adv"] is True
        assert g["kept"] is True and g["kept_for_update"] is False
    assert prov(skip)["final_param_checksum"] == after_first, "a skipped step must leave the weights where they were"

    _force_advantages(monkeypatch, G)
    mom = tmp_path / "momentum"
    assert TR.main(base_argv(tmp_path, mom, steps=2)) == 0
    second_m = records(mom, "train")[1]
    assert second_m["opt_step_taken"] is True and second_m["grad_norm_preclip"] == 0.0
    assert second_m["n_update_groups"] == G == second_m["n_zero_adv_groups"]
    assert prov(mom)["final_param_checksum"] != after_first, "a zero-gradient AdamW step must move the weights"
    # The weights agreed after step one, so both runs sampled the same step-two completions.
    assert [g["n_tokens"] for g in second["groups"]] == [g["n_tokens"] for g in second_m["groups"]]
    assert [g["strict"] for g in second["groups"]] == [g["strict"] for g in second_m["groups"]]


def test_skip_zero_adv_removes_only_the_zero_groups_from_the_update(tmp_path, monkeypatch):
    G = RT.GROUPS_PER_STEP
    _force_advantages(monkeypatch, G + 2)
    out = tmp_path / "partial"
    assert TR.main(base_argv(tmp_path, out, steps=2, extra=["--skip-zero-adv"])) == 0
    second = records(out, "train")[1]
    assert [g["zero_adv"] for g in second["groups"]] == [False, False] + [True] * (G - 2)
    assert [g["kept_for_update"] for g in second["groups"]] == [True, True] + [False] * (G - 2)
    assert second["n_kept_groups"] == G and second["n_zero_adv_groups"] == G - 2 and second["n_update_groups"] == 2
    assert second["opt_step_taken"] is True and second["grad_norm_preclip"] > 0


def test_the_skip_composes_with_the_verdict_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(TR, "score_texts", _parity_verdicts)
    out = tmp_path / "filtered"
    assert TR.main(base_argv(tmp_path, out, steps=2, arm="vfilter", extra=["--skip-zero-adv"])) == 0
    for rec in records(out, "train"):
        for g in rec["groups"]:
            v = [f == 0 for f in g["F"]]
            assert g["kept"] is (0 < sum(v) < len(v))
            assert g["kept_for_update"] is (g["kept"] and not g["zero_adv"])
        assert rec["n_update_groups"] == sum(g["kept_for_update"] for g in rec["groups"])
        assert rec["opt_step_taken"] is (rec["n_update_groups"] > 0)


# ------------------------------------------------------------------ entropy


def test_token_entropy_matches_an_explicit_softmax_entropy_and_the_frozen_logprobs():
    tok, model = _tiny_model()
    groups = _groups(tok, model, n_prompts=1)
    seqs, mask = groups[0].seqs, groups[0].mask
    with torch.no_grad():
        tok_lp, ent = TR.token_logprobs_and_entropy(model, seqs, mask)
        want_lp = TSG.token_logprobs(model, seqs, mask)
        logits = model(seqs).logits[:, :-1, :].float()
        p = torch.softmax(logits, dim=-1)
        want_ent = -(p * torch.log(p)).sum(dim=-1) * mask
    assert torch.equal(tok_lp, want_lp)
    assert float((ent - want_ent).abs().max()) <= 1e-4
    assert bool((ent[~mask] == 0).all()) and float(ent[mask].min()) > 0


def test_a_version_2_run_logs_the_step_one_entropy_that_the_initial_weights_recompute(tmp_path):
    out = tmp_path / "ent"
    assert TR.main(base_argv(tmp_path, out, steps=1)) == 0
    rec = records(out, "train")[0]
    assert np.isfinite(rec["entropy_tok_mean"]) and rec["entropy_tok_mean"] > 0
    tok, model = _tiny_model()
    eos = TR.eos_ids_of(tok)
    cands = {c["pid"]: c for c in json.loads((tmp_path / "ds.json").read_text())["train_candidates"]}
    comps = [json.loads(x) for x in (out / "completions.jsonl").read_text().splitlines()]
    total, n_tok = 0.0, 0
    for g in rec["groups"]:
        prompt = TR.encode_prompt(tok, cands[g["pid"]]["prompt"])
        rows = [c for c in comps if c["phase"] == "train" and c["step"] == 1 and c["pid"] == g["pid"]]
        width = len(prompt) + max(len(c["token_ids"]) for c in rows)
        seqs = torch.tensor([prompt + c["token_ids"] + [tok.pad_token_id] * (width - len(prompt) - len(c["token_ids"]))
                             for c in rows])
        mask = TSG.completion_mask(seqs, len(prompt), eos)
        assert mask.sum(dim=1).tolist() == g["n_tokens"]
        with torch.no_grad():
            _, ent = TR.token_logprobs_and_entropy(model, seqs, mask)
        total += float(ent.sum())
        n_tok += int(mask.sum())
    assert rec["entropy_tok_mean"] == pytest.approx(total / n_tok, rel=1e-5)


# ------------------------------------------------------------------ version 1 is byte for byte what it was


def test_a_version_1_manifest_row_still_validates_and_resolves_without_version_2_fields():
    row = RT.read_manifest_row(HERE / "slurm" / "manifest_reasoning.csv", 1)
    out = RT.validate_arm_row(row)
    assert out["design_version"] == 1 and "lr" not in out and out["arm"] == "plain"
    assert tuple(row.keys()) == RT.ARM_HEADER
    for i in range(1, 16):
        assert RT.validate_arm_row(RT.read_manifest_row(HERE / "slurm" / "manifest_reasoning.csv", i))["design_version"] == 1


@pytest.mark.parametrize("case", sorted(GOLDEN_SHA256))
def test_a_version_1_tiny_run_writes_the_golden_bytes(tmp_path, monkeypatch, case):
    versions = _versions()
    if versions != GOLDEN_VERSIONS:
        pytest.skip(f"the golden hashes were recorded with {GOLDEN_VERSIONS}, and this machine has {versions}")
    if case == "mixed_plain":
        monkeypatch.setattr(TR, "score_texts", _parity_verdicts)
    write_tiny_design(tmp_path)
    out = tmp_path / case
    argv = ["--dataset", str(tmp_path / "ds.json"), "--selection", str(tmp_path / "sel.json"), "--out-dir", str(out),
            "--tiny", "--device", "cpu"] + GOLDEN_ARGV[case]
    assert TR.main(argv) == 0
    for name in ("metrics.jsonl", "completions.jsonl"):
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == GOLDEN_SHA256[case][name], name
    assert prov(out)["final_param_checksum"] == GOLDEN_SHA256[case]["final_param_checksum"]
    c = cfg(out)
    for key in ("design_version", "kl_beta", "skip_zero_adv", "fixedref_c", "reference_loaded"):
        assert key not in c
    for rec in records(out, "train"):
        for key in ("entropy_tok_mean", "kl_loss", "kl_seq_mean", "n_zero_adv_groups", "n_update_groups"):
            assert key not in rec
        for g in rec["groups"]:
            assert "zero_adv" not in g and "kept_for_update" not in g


def test_a_version_1_run_never_touches_the_version_2_code(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("version 1 reached version-2 code")

    for name in ("frozen_reference", "token_logprobs_and_entropy", "accumulate_gradients_v2",
                 "micro_batched_update_v2", "check_frozen_v2", "read_frozen_v2"):
        monkeypatch.setattr(TR, name, boom)
    out = tmp_path / "v1"
    assert TR.main(base_argv(tmp_path, out, steps=1, version=1)) == 0
    assert "design_version" not in prov(out)


@pytest.mark.parametrize("extra", [["--kl-beta", "0.02"], ["--skip-zero-adv"], ["--fixedref-c", "2.0"],
                                   ["--arm", "restored"], ["--kind", "pilot"]])
def test_a_version_1_run_refuses_the_version_2_flags(tmp_path, extra, capsys):
    out = tmp_path / "refused"
    assert TR.main(base_argv(tmp_path, out, steps=1, version=1, extra=extra)) == 2
    assert "design version 2" in capsys.readouterr().err
    assert not out.exists()


# ------------------------------------------------------------------ the frozen version-2 file


def test_a_version_2_run_refuses_without_the_frozen_file_and_with_a_stale_one(tmp_path, capsys):
    write_tiny_design(tmp_path)
    common = ["--dataset", str(tmp_path / "ds.json"), "--selection", str(tmp_path / "sel.json"), "--tiny",
              "--device", "cpu", "--steps", "1", "--L", "16", "--design-version", "2", "--mid-eval-every", "0"]
    out = tmp_path / "missing"
    missing = tmp_path / "configs" / "reasoning_frozen_v2.json"
    assert not missing.exists()
    assert TR.main(common + ["--out-dir", str(out), "--frozen-v2", str(missing)]) == 2
    err = capsys.readouterr().err
    assert "does not exist yet" in err and "package_reasoning_source.py" in err and "Nothing was trained" in err
    assert not out.exists()
    stale = write_frozen_v2(tmp_path / "stale.json", training="0" * 64)
    assert TR.main(common + ["--out-dir", str(out), "--frozen-v2", str(stale)]) == 2
    assert "training sources hash to" in capsys.readouterr().err and not out.exists()
    stale_r = write_frozen_v2(tmp_path / "stale_r.json", reasoning="1" * 64)
    assert TR.main(common + ["--out-dir", str(out), "--frozen-v2", str(stale_r)]) == 2
    assert "reasoning sources hash to" in capsys.readouterr().err and not out.exists()
    (tmp_path / "bad.json").write_text(json.dumps({"schema": "other", "training_sources_digest": "a" * 64,
                                                   "reasoning_sources_digest": "b" * 64}))
    assert TR.main(common + ["--out-dir", str(out), "--frozen-v2", str(tmp_path / "bad.json")]) == 2
    assert "schema" in capsys.readouterr().err
    good = write_frozen_v2(tmp_path / "good.json")
    assert TR.main(common + ["--out-dir", str(out), "--frozen-v2", str(good)]) == 0
    assert prov(out)["frozen_v2"] == json.loads(good.read_text())


def test_the_default_frozen_path_is_the_configs_file_the_experimenter_writes():
    assert TR.FROZEN_V2_PATH == HERE / "configs" / "reasoning_frozen_v2.json"
    assert TR.FROZEN_TRAINING_DIGEST == {1: "6afdaa1fa50c1126c3278aa33887d4be4ec47e5283097bda081bfa563e3300aa", 2: None}
    args = TR.build_arg_parser().parse_args(["--dataset", "d", "--selection", "s"])
    assert Path(args.frozen_v2) == TR.FROZEN_V2_PATH
    assert (args.design_version, args.kl_beta, args.skip_zero_adv, args.fixedref_c) == (1, 0.0, False, None)


# ------------------------------------------------------------------ manifest mode for version 2


def v2_row(ds_sha, sel_sha, **kw):
    row = {"task_id": "1", "kind": "pilot", "stage": "3", "size": "0.5b", "model_id": RT.MODELS["0.5b"]["model_id"],
           "model_revision": RT.MODELS["0.5b"]["revision"], "arm": "verdict_only", "form": "linear", "w": "2.0",
           "w_task": "0.0", "adv_variant": "grpo", "group_filter": "none", "seed": "0", "steps": "100",
           "L": "512", "micro_batch": "2", "dataset_sha256": ds_sha, "selection_sha256": sel_sha,
           "lr": "1e-06", "kl_beta": "0.0", "skip_zero_adv": "0", "fixedref_c": "0.0"}
    row.update(kw)
    return row


def write_manifest(path, rows, header=RT.ARM_HEADER_V2):
    path.write_text(",".join(header) + "\n" + "".join(",".join(r[k] for k in header) + "\n" for r in rows))


def test_manifest_mode_carries_the_version_2_levers_from_the_row_and_refuses_moved_flags(tmp_path, capsys):
    ds, ds_sha, sel, sel_sha = write_tiny_design(tmp_path, L=512, steps=100)
    man = tmp_path / "m.csv"
    write_manifest(man, [v2_row(ds_sha, sel_sha, lr="1e-05", kl_beta="0.02", skip_zero_adv="1")])
    frozen = write_frozen_v2(tmp_path / "frozen.json")
    base = ["--manifest", str(man), "--row", "1", "--dataset", str(ds), "--selection", str(sel),
            "--out-root", str(tmp_path / "out"), "--device", "cpu", "--tiny", "--frozen-v2", str(frozen)]
    args = TR.build_arg_parser().parse_args(base + ["--expect-stage", "3"])
    design = TR.resolve_design(args)
    assert design["design_version"] == 2 and design["kind"] == "pilot" and design["steps"] == 100
    assert (design["lr"], design["kl_beta"], design["skip_zero_adv"], design["fixedref_c"]) == (1e-5, 0.02, True, 0.0)
    assert design["mid_eval_every"] == RT.EVAL_EVERY
    assert TR.run_dir_for(args, design).name == "1_pilot_0.5b_verdict_only_s0"
    TR.check_time_limit("pilot", 60)
    with pytest.raises(TR.Refused):
        TR.check_time_limit("pilot", 59)
    # The row sets the levers, so a flag that moves them is refused before anything is written.
    for extra in (["--lr", "1e-6"], ["--kl-beta", "0.05"], ["--skip-zero-adv"], ["--fixedref-c", "1.0"]):
        assert TR.main(base + ["--expect-stage", "3"] + extra) == 2
        assert "flags must stay at their defaults" in capsys.readouterr().err
    assert TR.main(base + ["--expect-stage", "4"]) == 2
    assert "stage 3 row" in capsys.readouterr().err
    assert TR.main(base + ["--expect-stage", "3", "--time-limit-minutes", "59"]) == 2
    assert "at least 60 minutes" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    # A version-1 design never carries the levers, in manifest mode too.
    v1 = {k: v for k, v in v2_row(ds_sha, sel_sha, kind="smoke", stage="0", arm="plain", w_task="1.0",
                                   seed="100", steps="20", L="512").items() if k in RT.ARM_HEADER}
    write_manifest(tmp_path / "v1.csv", [v1], header=RT.ARM_HEADER)
    args1 = TR.build_arg_parser().parse_args([x if x != str(man) else str(tmp_path / "v1.csv") for x in base]
                                             + ["--expect-stage", "0"])
    d1 = TR.resolve_design(args1)
    assert "design_version" not in d1 and "kl_beta" not in d1


def test_a_restored_row_in_manifest_mode_resolves_c_and_the_study_row_name(tmp_path):
    ds, ds_sha, sel, sel_sha = write_tiny_design(tmp_path, L=512, steps=300)
    man = tmp_path / "m.csv"
    write_manifest(man, [v2_row(ds_sha, sel_sha, kind="study", stage="4", arm="restored", w_task="1.0",
                                adv_variant="grpo_fixedref", steps="300", fixedref_c="2.0", seed="2")])
    args = TR.build_arg_parser().parse_args(["--manifest", str(man), "--row", "1", "--dataset", str(ds),
                                             "--selection", str(sel), "--out-root", str(tmp_path / "out"),
                                             "--expect-stage", "4"])
    design = TR.resolve_design(args)
    assert design["arm"] == "restored" and design["adv_variant"] == "grpo_fixedref" and design["fixedref_c"] == 2.0
    assert TR.run_dir_for(args, design).name == "1_study_0.5b_restored_s2"


# ------------------------------------------------------------------ the early-finish guard


class _FakeClock:
    """A clock that advances a fixed number of seconds at every reading, so a run's own timings add up
    to a wall budget without any waiting."""

    def __init__(self, step=300.0):
        self.t, self.step = 1_000_000.0, step

    def time(self):
        self.t += self.step
        return self.t

    def strftime(self, *a, **k):
        import time
        return time.strftime(*a, **k)


def test_early_finish_needed_seconds_follows_the_definition():
    assert TR.early_finish_needed_seconds([], 100.0, False) is None
    assert TR.early_finish_needed_seconds([10.0, 20.0], 100.0, False) == pytest.approx(115.0)
    assert TR.early_finish_needed_seconds([10.0, 20.0], 100.0, True) == pytest.approx(215.0)


def test_the_guard_stops_training_runs_the_final_evaluations_and_records_the_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(TR, "time", _FakeClock(300.0))
    out = tmp_path / "early"
    steps = 10
    assert TR.main(base_argv(tmp_path, out, steps=steps, extra=["--time-limit-minutes", "120",
                                                                 "--early-finish-margin-seconds", "120"])) == 0
    p = prov(out)
    assert p["stopped_early"] is True and 1 <= p["steps_completed"] < steps
    done = p["steps_completed"]
    fin = p["early_finish"]
    assert fin["before_step"] == done + 1 and fin["remaining_seconds"] < fin["needed_seconds"]
    assert fin["margin_seconds"] == 120.0 and fin["full_eval_seconds"] > 0 and fin["mean_step_seconds"] > 0
    train = records(out, "train")
    assert [r["step"] for r in train] == list(range(1, done + 1))
    evals = records(out, "eval")
    assert [(e["step"], e["set"], e["samples"]) for e in evals] == [(0, "hard", 4), (0, "random", 2),
                                                                     (done, "hard", 4), (done, "random", 2)]
    assert (out / "DONE").exists()
    timing = json.loads((out / "timing.json").read_text())
    assert timing["steps_completed"] == done and len(timing["step_seconds"]) == done
    assert cfg(out)["time_limit_minutes"] == 120
    # The analysis sees the stop: the curves name it, and admission refuses the run for it by name.
    run = A.load_run(out)
    curves = A.run_curves(run)
    assert curves["stopped_early"] is True and curves["steps_completed"] == done and curves["steps_planned"] == steps
    assert curves["step"] == list(range(1, done + 1)) and len(curves["entropy_tok_mean"]) == done
    assert all(np.isfinite(x) for x in curves["entropy_tok_mean"]) and all(x is None for x in curves["kl_seq_mean"])
    row = {"steps": str(steps), "arm": "plain", "dataset_sha256": p["dataset_sha256"],
           "selection_sha256": p["selection_sha256"], "model_revision": p["model_revision"]}
    reasons = A.admit(run, row, {"reasoning_sources_digest": p["reasoning_sources_digest"]}, None)
    assert any("stopped_early" in r for r in reasons)
    listing = A.curves_report([tmp_path])
    assert [c["stopped_early"] for c in listing] == [True]


def test_a_run_that_fits_its_limit_is_not_stopped_and_a_version_1_run_ignores_the_guard(tmp_path, monkeypatch):
    out = tmp_path / "fits"
    assert TR.main(base_argv(tmp_path, out, steps=2, extra=["--time-limit-minutes", "120"])) == 0
    assert prov(out)["stopped_early"] is False and prov(out)["early_finish"] is None
    assert [r["step"] for r in records(out, "train")] == [1, 2]
    monkeypatch.setattr(TR, "time", _FakeClock(300.0))
    v1 = tmp_path / "v1"
    assert TR.main(base_argv(tmp_path, v1, steps=4, version=1, extra=["--time-limit-minutes", "120"])) == 0
    assert [r["step"] for r in records(v1, "train")] == [1, 2, 3, 4] and "stopped_early" not in prov(v1)


# ------------------------------------------------------------------ the analysis curves


def test_run_curves_of_a_version_1_run_hold_the_length_and_none_for_the_new_fields(tmp_path):
    out = tmp_path / "v1"
    assert TR.main(base_argv(tmp_path, out, steps=2, version=1)) == 0
    c = A.run_curves(A.load_run(out))
    assert c["design_version"] == 1 and c["stopped_early"] is False and c["steps_completed"] == 2
    assert c["completion_tokens_mean"] == [r["completion_tokens_mean"] for r in records(out, "train")]
    assert c["entropy_tok_mean"] == [None, None] and c["kl_seq_mean"] == [None, None]
    assert A.CURVE_FIELDS == ("completion_tokens_mean", "entropy_tok_mean", "kl_seq_mean")


def test_the_curves_flag_writes_a_separate_file_and_leaves_the_report_alone(tmp_path, monkeypatch):
    import make_reasoning_manifest as MM
    import random
    from reasoning_fixtures import fake_run
    DS, SEL_SHA, SRC = "d" * 64, "e" * 64, "f" * 64
    selection = {"candidates_sha256": DS, "L": 384,
                 "sizes": {"0.5b": {"status": "ok", "model_revision": RT.MODELS["0.5b"]["revision"], "short_threshold": 50.0},
                           "1.5b": {"status": "dropped"}}}
    rows = MM.study_rows(selection, {"selection_sha256": SEL_SHA, "S": 150, "sizes_kept": ["0.5b"]}, DS, SEL_SHA)
    (tmp_path / "m.csv").write_bytes(MM.to_csv(RT.ARM_HEADER, rows))
    runs = tmp_path / "runs" / "900"
    runs.mkdir(parents=True)
    rng = random.Random(0)
    for row in rows:
        fake_run(runs, row, rng, SRC)
    (tmp_path / "frozen.json").write_text(json.dumps({"reasoning_sources_digest": SRC}))
    (tmp_path / "sel.json").write_text(json.dumps(selection))
    logs = tmp_path / "logs"
    logs.mkdir()
    for row in rows:
        (logs / f"wo-i-s1-900_{row['task_id']}.out").write_text('{"event": "done"}\n')
    common = ["--manifest", str(tmp_path / "m.csv"), "--runs", str(tmp_path / "runs"), "--frozen", str(tmp_path / "frozen.json"),
              "--selection", str(tmp_path / "sel.json"), "--logs", str(logs)]
    assert A.main(common + ["--out", str(tmp_path / "a.json")]) == 0
    assert A.main(common + ["--out", str(tmp_path / "b.json"), "--curves", str(tmp_path / "curves.json")]) == 0
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    curves = json.loads((tmp_path / "curves.json").read_text())
    assert len(curves) == 15 and all(c["design_version"] == 1 and c["stopped_early"] is False for c in curves)
    assert all(len(c["completion_tokens_mean"]) == 150 for c in curves)
