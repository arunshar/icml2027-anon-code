"""These tests run the WO-I trainer and calibration on CPU with a tiny model and the real tokenizer.

The tokenizer comes from the pinned 0.5B snapshot in the local cache and is loaded with
local_files_only, so these tests never download anything. They skip when that snapshot is absent.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import calibrate_reasoning as CAL  # noqa: E402
import reasoning_task as RT  # noqa: E402
import rewards as RW  # noqa: E402
import select_reasoning_pool as SEL  # noqa: E402
import train_reasoning_grpo as TR  # noqa: E402
import train_speed_grpo as TSG  # noqa: E402
from reasoning_fixtures import write_tiny_design  # noqa: E402


def _has_tokenizer():
    try:
        TR.model_snapshot(RT.MODELS["0.5b"]["model_id"], RT.MODELS["0.5b"]["revision"])
        return True
    except TR.Refused:
        return False


pytestmark = pytest.mark.skipif(not _has_tokenizer(), reason="the pinned 0.5B tokenizer is not cached here")


def tiny_run(tmp: Path, name: str, mid_every: int, seed: int = 0, steps: int = 3, arm: str = "plain"):
    if not (tmp / "ds.json").exists():
        write_tiny_design(tmp)
    ds, sel, out = tmp / "ds.json", tmp / "sel.json", tmp / name
    rc = TR.main(["--dataset", str(ds), "--selection", str(sel), "--out-dir", str(out), "--tiny",
                  "--device", "cpu", "--steps", str(steps), "--L", "16", "--seed", str(seed), "--arm", arm,
                  "--mid-eval-every", str(mid_every)])
    assert rc == 0
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tiny")
    return {"a": tiny_run(tmp, "a", 1), "b": tiny_run(tmp, "b", 1), "c": tiny_run(tmp, "c", 0)}


def records(run, phase):
    return [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()
            if json.loads(x)["phase"] == phase]


def test_tiny_run_writes_every_file_and_every_group(runs):
    run = runs["a"]
    for name in ("DONE", "config.json", "provenance.json", "metrics.jsonl", "completions.jsonl", "timing.json"):
        assert (run / name).exists()
    train = records(run, "train")
    assert [r["step"] for r in train] == [1, 2, 3]
    for r in train:
        assert len(r["groups"]) == RT.GROUPS_PER_STEP
        for g in r["groups"]:
            assert len(g["n_tokens"]) == len(g["F"]) == len(g["adv"]) == len(g["rewards"]) == RT.GROUP_SIZE
            assert all(1 <= n <= 16 for n in g["n_tokens"])
    evals = records(run, "eval")
    assert [(r["step"], r["set"]) for r in evals] == [(0, "hard"), (0, "random"), (1, "hard"), (2, "hard"),
                                                     (3, "hard"), (3, "random")]
    prov = json.loads((run / "provenance.json").read_text())
    # Changed deliberately on 2026-11-06. This line used to assert that a fresh run records the frozen
    # version-1 digest, 6afdaa1fa50c1126c3278aa33887d4be4ec47e5283097bda081bfa563e3300aa, which is now
    # TR.FROZEN_TRAINING_DIGEST[1]. The training sources moved for the ICML 2027 blocks of November 2026,
    # so a new run records the CURRENT digest, and only the frozen version-1 runs carry the old value.
    assert prov["training_sources_digest"] == TSG.training_sources_digest()
    assert TR.FROZEN_TRAINING_DIGEST[1] == "6afdaa1fa50c1126c3278aa33887d4be4ec47e5283097bda081bfa563e3300aa"
    assert "design_version" not in prov and "stopped_early" not in prov, "a version-1 run gains no field"
    assert prov["reasoning_sources_digest"] == RT.reasoning_sources_digest()
    assert prov["model_snapshot"] == RT.MODELS["0.5b"]["revision"]
    # The S rule and the memory rule read timing.json only, so its fields are a contract.
    timing = json.loads((run / "timing.json").read_text())
    assert set(timing) == SEL.TIMING_KEYS and timing["steps_completed"] == 3
    n_completions = len((run / "completions.jsonl").read_text().splitlines())
    assert n_completions == 3 * 24 + (3 * 4 + 2 * 2) * 2 + 3 * 2


def test_two_runs_with_one_seed_write_identical_logs(runs):
    for name in ("metrics.jsonl", "completions.jsonl"):
        assert (runs["a"] / name).read_bytes() == (runs["b"] / name).read_bytes()
    assert json.loads((runs["a"] / "provenance.json").read_text())["final_param_checksum"] == \
        json.loads((runs["b"] / "provenance.json").read_text())["final_param_checksum"]


def train_completions(run):
    return [x for x in (run / "completions.jsonl").read_text().splitlines() if json.loads(x)["phase"] == "train"]


def test_evaluation_restores_the_training_random_state(runs):
    # Run c skips the middle evaluations, and its training groups are still identical. The tiny model
    # never samples an end token, so every logged group looks alike, and the sampled token ids are
    # the part of the log that shows whether the training random state was restored.
    assert records(runs["a"], "train") == records(runs["c"], "train")
    assert train_completions(runs["a"]) == train_completions(runs["c"])
    assert len(set(train_completions(runs["a"]))) > 1


def _parity_verdicts(records_, reference):
    """Score the real parses, then set F from the parity of the token ids so that groups are mixed.

    The random tiny model never writes a boxed answer, so without this every group is all-fail and
    every advantage is zero, and the update, the filter and the gate would never be exercised."""
    TR_score_texts(records_, reference)
    for r in records_:
        r["F"] = sum(r["token_ids"]) % 2


TR_score_texts = TR.score_texts


@pytest.fixture(scope="module")
def mixed_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mixed")
    write_tiny_design(tmp)
    mp = pytest.MonkeyPatch()
    mp.setattr(TR, "score_texts", _parity_verdicts)
    out = {}
    try:
        # The version-1 arms only. RT.ARMS also holds the restored arm of design version 2 (ICML 2027,
        # November 2026), which a version-1 run refuses, and tests/test_reasoning_v2.py covers it.
        for arm in RT.STUDY_ARMS["0.5b"]:
            d = tmp / arm
            rc = TR.main(["--dataset", str(tmp / "ds.json"), "--selection", str(tmp / "sel.json"), "--out-dir",
                          str(d), "--tiny", "--device", "cpu", "--steps", "3", "--L", "16", "--seed", "0",
                          "--arm", arm, "--mid-eval-every", "0"])
            assert rc == 0
            out[arm] = d
    finally:
        mp.undo()
    return out


def test_every_arm_logs_what_its_own_rule_gives_and_passes_the_w0_replay(mixed_runs):
    import analyze_reasoning as A
    import rewards as RW
    for arm, d in mixed_runs.items():
        run = A.load_run(d)
        report = A.score_w0([run])
        assert report["holds"], (arm, report)
        groups = [g for rec in run["train"] for g in rec["groups"]]
        mixed = [g for g in groups if not g["all_fail"] and not g["all_pass"]]
        assert mixed, arm
        for rec in run["train"]:
            assert rec["opt_step_taken"] == any(g["kept"] for g in rec["groups"])
            assert rec["n_kept_groups"] == sum(bool(g["kept"]) for g in rec["groups"])
        for g in groups:
            assert g["kept"] == RW.keep_group(RT.ARMS[arm]["group_filter"], g["rewards"], [f == 0 for f in g["F"]])
            if arm == "vfilter":
                assert g["kept"] == (not g["all_fail"] and not g["all_pass"])
    # The plain arm takes real updates, so its final weights differ from the weights it started from.
    assert any(r["grad_norm_preclip"] for r in records(mixed_runs["plain"], "train"))
    _, start = _tiny_model()
    plain = json.loads((mixed_runs["plain"] / "provenance.json").read_text())
    assert plain["final_param_checksum"] != TSG.param_checksum(start)


def _tiny_model():
    TSG.seed_everything(0)
    tok, model, _ = TR.load_model(RT.MODELS["0.5b"]["model_id"], RT.MODELS["0.5b"]["revision"], "cpu", tiny=True)
    model.eval()
    return tok, model


def test_padded_generation_matches_unpadded_scoring():
    tok, model = _tiny_model()
    prompts = [TR.encode_prompt(tok, RT.prompt_text("What is 2 plus 3?")),
               TR.encode_prompt(tok, RT.prompt_text("A much longer question about apples, pears and "
                                                    "several other fruits that a farmer sells at a market?"))]
    assert len(prompts[0]) != len(prompts[1])
    n = 3
    groups, out, width = TR.batched_generate(model, tok, prompts, n, TR.Sampler(), 10, "cpu", return_logits=True)
    logits = torch.stack(out.logits, dim=1)
    eos = TR.eos_ids_of(tok)
    worst = 0.0
    for i, (seqs, plen) in enumerate(groups):
        G = seqs.shape[1] - plen
        mask = TSG.completion_mask(seqs, plen, eos)
        with torch.no_grad():
            scored = TSG.token_logprobs(model, seqs, mask)[:, plen - 1:plen - 1 + G]
        gen = torch.log_softmax(logits[i * n:(i + 1) * n].float(), dim=-1)
        gen = gen.gather(-1, seqs[:, plen:].unsqueeze(-1)).squeeze(-1) * mask[:, plen - 1:plen - 1 + G]
        worst = max(worst, float((scored - gen).abs().max()))
    assert worst <= 1e-5


class _NoStep:
    def __init__(self, model):
        self.model = model

    def zero_grad(self):
        self.model.zero_grad()

    def step(self):
        pass


@pytest.mark.parametrize("kept", [(True, True), (True, False)])
def test_micro_batched_gradient_equals_the_frozen_update(kept):
    tok, model = _tiny_model()
    prompts = [TR.encode_prompt(tok, RT.prompt_text(q)) for q in ("What is 7 minus 2?", "How many legs do 3 cats have?")]
    gen = TR.batched_generate(model, tok, prompts, RT.GROUP_SIZE, TR.Sampler(), 12, "cpu")
    eos = TR.eos_ids_of(tok)
    masks = [TSG.completion_mask(s, p, eos) for s, p in gen]
    g = torch.Generator().manual_seed(3)
    adv = [torch.randn(RT.GROUP_SIZE, generator=g) for _ in gen]
    frozen, micro = copy.deepcopy(model).train(), copy.deepcopy(model).train()
    args = SimpleNamespace(kl_beta=0.0, clip_coef=0.2, grad_clip=1e9, backward_per_group=False)
    TSG.policy_update(frozen, None, _NoStep(frozen), [s for s, _ in gen], masks, adv, list(kept), args)
    micro.zero_grad()
    groups = [TR.Group(f"p{i}", s, m, a, k) for i, ((s, _), m, a, k) in enumerate(zip(gen, masks, adv, kept))]
    stats = TR.accumulate_gradients(micro, groups, RT.MICRO_BATCH, 0.2)
    assert stats["n_kept_groups"] == sum(kept)
    worst = max(float((a.grad - b.grad).abs().max()) for a, b in zip(frozen.parameters(), micro.parameters()))
    assert worst <= 1e-6
    assert max(float(p.grad.abs().max()) for p in micro.parameters()) > 0


def test_no_kept_group_takes_no_optimizer_step():
    tok, model = _tiny_model()
    before = TSG.param_checksum(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    seqs = torch.tensor([[1, 2, 3, 4]] * 4)
    mask = torch.ones(4, 3, dtype=torch.bool)
    stats = TR.micro_batched_update(model, opt, [TR.Group("p", seqs, mask, torch.ones(4), False)], 2, 0.2, 1.0)
    assert stats["opt_step_taken"] is False and TSG.param_checksum(model) == before


# ------------------------------------------------------------------ manifest mode refusals


def arm_row(ds_sha, sel_sha, **kw):
    row = {"task_id": "1", "kind": "smoke", "stage": "0", "size": "0.5b", "model_id": RT.MODELS["0.5b"]["model_id"],
           "model_revision": RT.MODELS["0.5b"]["revision"], "arm": "plain", "form": "linear", "w": "2.0",
           "w_task": "1.0", "adv_variant": "grpo", "group_filter": "none", "seed": "100", "steps": "20",
           "L": "256", "micro_batch": "2", "dataset_sha256": ds_sha, "selection_sha256": sel_sha}
    row.update(kw)
    return row


def write_manifest(path, rows):
    header = RT.ARM_HEADER
    path.write_text(",".join(header) + "\n" + "".join(",".join(r[k] for k in header) + "\n" for r in rows))


@pytest.fixture()
def manifest_design(tmp_path):
    ds, ds_sha, sel, sel_sha = write_tiny_design(tmp_path, L=256, steps=20)
    man = tmp_path / "m.csv"
    write_manifest(man, [arm_row(ds_sha, sel_sha)])
    base = ["--manifest", str(man), "--row", "1", "--dataset", str(ds), "--selection", str(sel),
            "--out-root", str(tmp_path / "out"), "--device", "cpu", "--tiny"]
    return tmp_path, base, ds_sha, sel_sha


def test_manifest_mode_refuses_before_training(manifest_design, capsys):
    tmp, base, ds_sha, sel_sha = manifest_design
    assert TR.main(base + ["--expect-stage", "1"]) == 2
    assert "stage 0 row" in capsys.readouterr().err
    assert TR.main(base + ["--expect-stage", "0", "--time-limit-minutes", "59"]) == 2
    assert "at least 60 minutes" in capsys.readouterr().err
    assert TR.main(base + ["--expect-stage", "0", "--require-gpu-name", RT.REQUIRED_GPU_NAME]) == 2
    assert "the device is" in capsys.readouterr().err
    write_manifest(tmp / "bad.csv", [arm_row("0" * 64, sel_sha)])
    bad = [x if x != str(tmp / "m.csv") else str(tmp / "bad.csv") for x in base]
    assert TR.main(bad + ["--expect-stage", "0"]) == 2
    assert "SHA-256" in capsys.readouterr().err
    write_manifest(tmp / "odd.csv", [arm_row(ds_sha, sel_sha, micro_batch="1")])
    odd = [x if x != str(tmp / "m.csv") else str(tmp / "odd.csv") for x in base]
    assert TR.main(odd + ["--expect-stage", "0"]) == 2
    assert "micro-batch" in capsys.readouterr().err
    assert not (tmp / "out").exists()


def test_manifest_mode_resolves_every_field_from_the_row(manifest_design):
    tmp, base, ds_sha, sel_sha = manifest_design
    args = TR.build_arg_parser().parse_args(base + ["--expect-stage", "0"])
    design = TR.resolve_design(args)
    assert design["mode"] == "manifest" and design["kind"] == "smoke" and design["steps"] == 20
    assert design["mid_eval_every"] == RT.SMOKE_MID_EVAL_EVERY and design["micro_batch"] == 2
    assert design["dataset_sha256"] == ds_sha and design["manifest_row"]["L"] == "256"
    assert TR.run_dir_for(args, design).name == "1_smoke_0.5b_plain_s100"
    TR.check_time_limit("smoke", 60)
    with pytest.raises(TR.Refused):
        TR.check_time_limit("study", 119)


# ------------------------------------------------------------------ calibration


def cal_args(tmp, root, **kw):
    argv = ["--dataset", str(tmp / "ds.json"), "--out-root", str(root), "--tiny", "--device", "cpu",
            "--limit-problems", "6", "--chunk-problems", "2", "--samples", "3", "--max-new-tokens", "12"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def test_calibration_writes_as_it_goes_and_resumes_with_the_same_draws(tmp_path):
    write_tiny_design(tmp_path)
    assert CAL.main(cal_args(tmp_path, tmp_path / "full")) == 0
    full = (tmp_path / "full" / "calibration" / "0_0.5b_train" / "records.jsonl").read_text()
    assert CAL.main(cal_args(tmp_path, tmp_path / "cut", stop_after_chunks=1)) == 0
    cut_dir = tmp_path / "cut" / "calibration" / "0_0.5b_train"
    assert len(cut_dir.joinpath("records.jsonl").read_text().splitlines()) == 2
    assert not (cut_dir / "DONE").exists()
    with cut_dir.joinpath("records.jsonl").open("a") as fh:   # this line imitates a chunk cut off mid-write
        fh.write('{"pid": "tr2", "sam')
    assert CAL.main(cal_args(tmp_path, tmp_path / "cut")) == 0
    assert cut_dir.joinpath("records.jsonl").read_text() == full
    assert (cut_dir / "DONE").exists()
    prov = json.loads((cut_dir / "provenance.json").read_text())
    assert [a["resumed_at_problem"] for a in prov["attempts"]] == [0, 2]
    lines = [json.loads(x) for x in full.splitlines()]
    assert [x["pid"] for x in lines] == [f"tr{i}" for i in range(6)]
    for x in lines:
        assert len(x["samples"]) == 3
        for s in x["samples"]:
            assert set(s["by_L"]) == {"256", "384", "512"} and 1 <= s["n_tokens"] <= 12
    # A finished task returns at once, before any model is loaded.
    assert CAL.main(cal_args(tmp_path, tmp_path / "cut")) == 0


def test_calibration_manifest_rows_are_validated(tmp_path, capsys):
    write_tiny_design(tmp_path)
    man = tmp_path / "cal.csv"
    row = {"task_id": "1", "kind": "calibration", "stage": "0", "size": "0.5b",
           "model_id": RT.MODELS["0.5b"]["model_id"], "model_revision": RT.MODELS["0.5b"]["revision"],
           "split": "train", "samples": "4", "max_new_tokens": "512", "chunk_problems": "8", "seed": "0",
           "dataset_sha256": "0" * 64}
    man.write_text(",".join(RT.CALIBRATION_HEADER) + "\n" + ",".join(row[k] for k in RT.CALIBRATION_HEADER) + "\n")
    assert CAL.main(["--manifest", str(man), "--row", "1", "--dataset", str(tmp_path / "ds.json"),
                     "--out-root", str(tmp_path / "o"), "--device", "cpu"]) == 2
    assert "8 completions" in capsys.readouterr().err
    assert CAL.main(["--dataset", str(tmp_path / "ds.json"), "--out-root", str(tmp_path / "o"),
                     "--time-limit-minutes", "90"]) == 2
    assert "at least 120 minutes" in capsys.readouterr().err


def test_a_missing_model_is_refused_before_any_directory_is_written(tmp_path, capsys):
    ds, ds_sha, sel, sel_sha = write_tiny_design(tmp_path, L=256, steps=20)
    absent = "1" * 40
    write_manifest(tmp_path / "m.csv", [arm_row(ds_sha, sel_sha, model_revision=absent)])
    assert TR.main(["--manifest", str(tmp_path / "m.csv"), "--row", "1", "--dataset", str(ds),
                    "--selection", str(sel), "--out-root", str(tmp_path / "out"), "--device", "cpu",
                    "--expect-stage", "0"]) == 2
    assert "not in the local cache" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    row = {"task_id": "2", "kind": "calibration", "stage": "0", "size": "0.5b",
           "model_id": RT.MODELS["0.5b"]["model_id"], "model_revision": absent, "split": "test",
           "samples": "8", "max_new_tokens": "512", "chunk_problems": "8", "seed": "0", "dataset_sha256": ds_sha}
    (tmp_path / "cal.csv").write_text(",".join(RT.CALIBRATION_HEADER) + "\n"
                                      + ",".join(row[k] for k in RT.CALIBRATION_HEADER) + "\n")
    assert CAL.main(["--manifest", str(tmp_path / "cal.csv"), "--row", "2", "--dataset", str(ds),
                     "--out-root", str(tmp_path / "cal_out"), "--device", "cpu"]) == 2
    assert "not in the local cache" in capsys.readouterr().err
    assert not (tmp_path / "cal_out").exists()


# ------------------------------------------------------------------ review additions


def test_the_evaluation_draws_do_not_depend_on_the_training_random_state():
    import io
    tok, model = _tiny_model()
    from reasoning_fixtures import tiny_dataset
    cands = {c["pid"]: c for c in tiny_dataset()["test_candidates"]}
    outs = []
    for seed in (1, 2):
        TSG.seed_everything(seed)
        expected_next = torch.rand(3)
        TSG.seed_everything(seed)
        fh = io.StringIO()
        res = TR.evaluate(model, tok, "cpu", [("hard", ["te0", "te1", "te2"], 2)], cands, TR.eos_ids_of(tok), 8,
                          TR.Sampler(), RT.EVAL_SEED, 4, None, 0, fh, {})
        # The training random state is restored, so the next training draw is unchanged.
        assert torch.equal(torch.rand(3), expected_next)
        outs.append((res, fh.getvalue()))
    assert outs[0] == outs[1]


def test_a_row_whose_L_differs_from_the_selection_is_refused(manifest_design, capsys):
    tmp, base, ds_sha, sel_sha = manifest_design
    write_manifest(tmp / "l.csv", [arm_row(ds_sha, sel_sha, L="384")])
    args = [x if x != str(tmp / "m.csv") else str(tmp / "l.csv") for x in base]
    assert TR.main(args + ["--expect-stage", "0"]) == 2
    assert "L differs" in capsys.readouterr().err
    assert not (tmp / "out").exists()


def test_calibration_scores_each_L_on_the_completion_cut_at_L():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TR.model_snapshot(RT.MODELS["0.5b"]["model_id"],
                                                              RT.MODELS["0.5b"]["revision"])), local_files_only=True)
    eos = TR.eos_ids_of(tok)
    prompt = TR.encode_prompt(tok, RT.prompt_text("What is 3 plus 4?"))
    filler = tok(" one" * 300).input_ids
    box = tok(" so the answer is \\boxed{7}.").input_ids
    comp = filler + box + [tok.convert_tokens_to_ids("<|im_end|>")]
    assert 300 <= len(filler) < 380 and len(comp) < 400
    seqs = torch.tensor([prompt + comp])
    line = CAL.problem_line(tok, "tr0", 0, 0, "7", seqs, len(prompt), eos)
    s = line["samples"][0]
    assert s["n_tokens"] == len(comp) and s["finished"] and s["strict"] == "7"
    # The box sits after token 256, so the completion cut at 256 has no answer, and the longer cuts do.
    assert s["by_L"]["256"] == {"strict": None, "F": 1}
    assert s["by_L"]["384"] == {"strict": "7", "F": 0} and s["by_L"]["512"] == {"strict": "7", "F": 0}


def test_a_resumed_calibration_keeps_whole_chunks_only(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps({"pid": f"tr{i}"}) + "\n" for i in range(5)))
    assert [x["pid"] for x in CAL.read_finished(path, 2)] == ["tr0", "tr1", "tr2", "tr3"]
    assert [x["pid"] for x in CAL.read_finished(path, 1)] == [f"tr{i}" for i in range(5)]
    assert CAL.read_finished(tmp_path / "absent.jsonl", 2) == []


# ------------------------------------------------------------------ fixes approved on 23 September


class _StubTok:
    """This tokenizer stand-in has the two ids that batched_generate reads, and the end ids are 0 and 9."""
    pad_token_id = 0
    eos_token_id = 9

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


class _EndsAt:
    """This stand-in for a model replays fixed completion lengths through batched_generate.

    Row r of the batch gets ends[r] new tokens, and its last token is the end token given for it.
    A row whose length is None never ends and runs to max_new_tokens. Every row is padded after its
    end, and the batch ends at its longest row, as a real generate call does."""

    def __init__(self, ends, end_tokens):
        self.ends, self.end_tokens = ends, end_tokens

    def generate(self, input_ids, attention_mask=None, num_return_sequences=1, pad_token_id=None,
                 max_new_tokens=None, return_dict_in_generate=True, output_logits=False, **_):
        rows = input_ids.repeat_interleave(num_return_sequences, dim=0)
        lengths = [max_new_tokens if e is None else e for e in self.ends]
        new = torch.full((rows.shape[0], max(lengths)), pad_token_id, dtype=torch.long)
        for r, (k, e) in enumerate(zip(lengths, self.ends)):
            new[r, :k] = torch.arange(100, 100 + k)
            if e is not None:
                new[r, k - 1] = self.end_tokens[r]
        return SimpleNamespace(sequences=torch.cat([rows, new], dim=1))


def test_each_group_is_trimmed_to_its_own_longest_completion():
    # The first group ends after at most 5 tokens, and one row of the second group never ends. The
    # batch is therefore 8 + 12 columns wide, and the first group keeps only 5 + 5 of them. One row
    # ends on the pad id, which is also an end id.
    prompts = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8]]
    ends = [2, 5, 3, 1, None, 4, 6, 2]
    stub = _EndsAt(ends, [9, 9, 9, 0, 9, 9, 0, 9])
    groups = TR.batched_generate(stub, _StubTok(), prompts, 4, TR.Sampler(), 12, "cpu")
    eos = TR.eos_ids_of(_StubTok())
    assert eos == [0, 9]
    widths = [seqs.shape[1] for seqs, _ in groups]
    assert widths == [5 + 5, 8 + 12]
    assert widths[0] < 8 + 12 - (8 - 5)  # the first group's width without the trim
    counts = [[r["n_tokens"] for r in TR.completion_records(_StubTok(), seqs, plen, eos)[0]]
              for seqs, plen in groups]
    assert counts == [[2, 5, 3, 1], [12, 4, 6, 2]]
    for seqs, plen in groups:
        assert TR.group_width(seqs, plen, eos) == seqs.shape[1]
    # Here the shorter prompt's group ends within the 3 columns of left padding that the batch adds to
    # it. Its width must still count from its own prompt, so it keeps 5 + 2 columns and not 8 + 1.
    stub = _EndsAt([1, 2, 1, 2, 12, 3, 3, 3], [9] * 8)
    groups = TR.batched_generate(stub, _StubTok(), prompts, 4, TR.Sampler(), 12, "cpu")
    assert [seqs.shape[1] for seqs, _ in groups] == [5 + 2, 8 + 12]


def _padded_group(tok, prompt, ends, width, seed=1):
    """Return a group of sequences that end at the given lengths and carry padding up to width."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for k in ends:
        comp = torch.randint(10, 5000, (width - len(prompt),), generator=g).tolist()
        comp[k - 1] = tok.eos_token_id
        comp[k:] = [tok.pad_token_id] * (len(comp) - k)
        rows.append(prompt + comp)
    return torch.tensor(rows)


def test_trimming_keeps_log_probabilities_masks_counts_rewards_and_gradients():
    tok, model = _tiny_model()
    eos = TR.eos_ids_of(tok)
    prompt = TR.encode_prompt(tok, RT.prompt_text("What is 2 plus 3?"))
    plen = len(prompt)
    # The batch is 20 completion tokens wide, and this group's longest completion has 7 tokens.
    full = _padded_group(tok, prompt, [3, 7, 5, 2], plen + 20)
    trimmed = TR.trim_group(full, plen, eos)
    assert trimmed.shape[1] == plen + 7 < full.shape[1]
    rec_full, mask_full = TR.completion_records(tok, full, plen, eos)
    rec_trim, mask_trim = TR.completion_records(tok, trimmed, plen, eos)
    assert rec_full == rec_trim and [r["n_tokens"] for r in rec_trim] == [3, 7, 5, 2]
    assert torch.equal(mask_full[:, :mask_trim.shape[1]], mask_trim)
    assert not bool(mask_full[:, mask_trim.shape[1]:].any())
    fails = [1, 0, 1, 1]
    n_tok = [r["n_tokens"] for r in rec_trim]
    r = RT.reasoning_rewards(fails, n_tok, "linear", RT.W, 1.0, 16)
    a = torch.tensor(RW.advantage_variant(r, "grpo"), dtype=torch.float32)
    with torch.no_grad():
        lp_full = TSG.seq_logprobs(model, full, mask_full)
        lp_trim = TSG.seq_logprobs(model, trimmed, mask_trim)
    # The narrower forward pass can change float32 rounding, so the check allows a rounding gap. A
    # dropped scored token would move a summed log-probability by several nats.
    assert float((lp_full - lp_trim).abs().max()) <= 1e-4
    grads = []
    for seqs, mask in ((full, mask_full), (trimmed, mask_trim)):
        m = copy.deepcopy(model).train()
        m.zero_grad()
        TR.accumulate_gradients(m, [TR.Group("p", seqs, mask, a, True)], RT.MICRO_BATCH, 0.2)
        grads.append([p.grad.clone() for p in m.parameters()])
    worst = max(float((x - y).abs().max()) for x, y in zip(*grads))
    assert worst <= 1e-6
    assert max(float(x.abs().max()) for x in grads[1]) > 0


def _early_end_loader(real_load):
    """Wrap load_model so that every generate call ends row r after 1 + (5 r mod T) tokens, where T is
    the number of generated columns. The tiny random model almost never samples an end token, so this
    gives each group of four its own longest completion inside one batch."""
    def load(*a, **kw):
        tok, model, info = real_load(*a, **kw)
        generate = model.generate

        def early_ends(input_ids, **gkw):
            out = generate(input_ids, **gkw)
            seqs, w = out.sequences.clone(), input_ids.shape[1]
            T = seqs.shape[1] - w
            ends = [1 + (5 * r) % T for r in range(seqs.shape[0])]
            for r, e in enumerate(ends):
                seqs[r, w + e - 1] = tok.eos_token_id
                seqs[r, w + e:] = tok.pad_token_id
            out.sequences = seqs[:, :w + max(ends)]
            return out

        model.generate = early_ends
        return tok, model, info
    return load


def test_the_trainer_scores_each_group_at_its_own_width(tmp_path, monkeypatch):
    import analyze_reasoning as A
    ds, _, sel, _ = write_tiny_design(tmp_path)
    seen = []
    real_update = TR.micro_batched_update

    def spy(model, opt, groups, *rest):
        seen.append([(g.pid, g.seqs.shape[1], g.mask.shape[1]) for g in groups])
        return real_update(model, opt, groups, *rest)

    monkeypatch.setattr(TR, "load_model", _early_end_loader(TR.load_model))
    monkeypatch.setattr(TR, "micro_batched_update", spy)
    out = tmp_path / "run"
    assert TR.main(["--dataset", str(ds), "--selection", str(sel), "--out-dir", str(out), "--tiny", "--device",
                    "cpu", "--steps", "2", "--L", "16", "--seed", "0", "--arm", "plain",
                    "--mid-eval-every", "0"]) == 0
    tok, _ = _tiny_model()
    cands = {c["pid"]: c for c in json.loads(ds.read_text())["train_candidates"]}
    train = records(out, "train")
    assert len(seen) == len(train) == 2
    trimmed = 0
    for rec, widths in zip(train, seen):
        step_longest = max(n for g in rec["groups"] for n in g["n_tokens"])
        for j, (g, (pid, w, mw)) in enumerate(zip(rec["groups"], widths)):
            assert g["pid"] == pid
            # The logged lengths are the injected ones, so the trim changed no token count.
            assert g["n_tokens"] == [1 + (5 * (4 * j + i)) % 16 for i in range(4)]
            plen = len(TR.encode_prompt(tok, cands[pid]["prompt"]))
            assert w == plen + max(g["n_tokens"]) and mw == w - 1
            trimmed += max(g["n_tokens"]) < step_longest
    # Four of the six groups end before the step's longest completion, so a batch-wide width fails here.
    assert trimmed == 2 * 4
    assert A.score_w0([A.load_run(out)])["holds"]


def _bundle_dir(root: Path, data):
    """Write the files of one real bundle into a directory, as the cluster extracts it.

    Built as a version-2 bundle since 2026-11-06, because the training sources no longer hash to the
    version-1 digest and a version-1 bundle refuses to build from this tree."""
    PK = pytest.importorskip("package_reasoning_source",
                             reason="the bundle builder is not part of the anonymous code archive")
    bundle = PK.build(PK.HERE, data, design_version=RT.DESIGN_VERSION_V2)
    d = root / bundle["name"]
    for name, raw in bundle["files"].items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(raw)
    return d, bundle


@pytest.mark.parametrize("retry_from", ["same bundle", "other bundle", "other sources", "other design"])
def test_a_calibration_retry_keeps_the_first_digests_and_records_its_own(tmp_path, monkeypatch, retry_from):
    write_tiny_design(tmp_path)
    dataset, manifest = "configs/reasoning_dataset.json", "slurm/manifest_reasoning_calibration.csv"
    a_dir, a = _bundle_dir(tmp_path / "a", [dataset, manifest])
    b_dir, b = _bundle_dir(tmp_path / "b", [dataset])
    source = RT.reasoning_sources_digest()
    assert a["source_digest"] == b["source_digest"] == source and a["bundle_digest"] != b["bundle_digest"]
    assert CAL.bundle_digest(a_dir) == a["bundle_digest"] and CAL.bundle_digest(tmp_path) is None
    monkeypatch.setattr(CAL, "HERE", a_dir)
    assert CAL.main(cal_args(tmp_path, tmp_path / "o", stop_after_chunks=1)) == 0
    if retry_from == "other bundle":
        monkeypatch.setattr(CAL, "HERE", b_dir)
    if retry_from == "other sources":
        monkeypatch.setattr(CAL.RT, "reasoning_sources_digest", lambda *_args, **_kw: "e" * 64)
    retry_args = {"seed": 5} if retry_from == "other design" else {}
    assert CAL.main(cal_args(tmp_path, tmp_path / "o", **retry_args)) == 0
    run = tmp_path / "o" / "calibration" / "0_0.5b_train"
    assert (run / "DONE").exists()
    prov = json.loads((run / "provenance.json").read_text())
    first, retry = prov["attempts"]
    assert [first["resumed_at_problem"], retry["resumed_at_problem"]] == [0, 2]
    for attempt in (first, retry):
        assert set(RT.CAL_ATTEMPT_FIELDS) <= set(attempt)
    # The top of the file keeps the first attempt's values, whatever the retry ran from.
    assert all(prov[k] == first[k] for k in RT.CAL_ATTEMPT_FIELDS)
    assert (first["bundle_digest"], first["reasoning_sources_digest"]) == (a["bundle_digest"], source)
    if retry_from == "same bundle":
        assert all(retry[k] == first[k] for k in RT.CAL_ATTEMPT_FIELDS)
        SEL.check_attempts(prov, "this task")
    else:
        field = {"other bundle": "bundle_digest", "other sources": "reasoning_sources_digest",
                 "other design": "design"}[retry_from]
        assert retry[field] != first[field]
        expected = {"other bundle": b["bundle_digest"], "other sources": "e" * 64,
                    "other design": dict(first["design"], seed=5)}[retry_from]
        assert retry[field] == expected
        with pytest.raises(SEL.Stop, match=f"different values of {field}$"):
            SEL.check_attempts(prov, "this task")
