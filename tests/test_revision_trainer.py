"""Exercise the strict trainer's budget, frozen rollout, and auditable completion path."""
import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import revision_task as task
import train_revision as trainer
from revision_ppo import PromptValueCritic


class TinyActor(torch.nn.Module):
    """A one-token categorical actor lets the tests exercise actual gradients."""

    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.1, 0.0]))
        self.forward_modes = []

    def forward(self, ids):
        self.forward_modes.append(self.training)
        return SimpleNamespace(logits=self.logits.expand(ids.shape[0], ids.shape[1], -1))

    def generate(self, ids, num_return_sequences, **kwargs):
        assert kwargs["temperature"] == kwargs["top_p"] == kwargs["repetition_penalty"] == 1
        assert kwargs["top_k"] == 0
        token = torch.multinomial(self.logits.softmax(0), num_return_sequences, replacement=True)
        return torch.cat([ids.expand(num_return_sequences, -1), token[:, None]], dim=1)


class TinyTokenizer:
    eos_token_id = None
    pad_token_id = None

    def decode(self, ids, **kwargs):
        return ["[]", "[0]", "not JSON", "[15]"][int(ids[0])]


@pytest.fixture(scope="module")
def dataset():
    return task.build_dataset(heldout_count=2)


def _args(tmp_path, dataset, method="ppo", extra=()):
    data_path = tmp_path / "data.json"
    task.save_manifest(dataset, data_path)
    args = trainer.build_arg_parser().parse_args([
        "--dataset", str(data_path), "--out", str(tmp_path / "run"),
        "--method", method, "--weight", "25", "--reward-scale", "26",
        "--steps", "2", "--group-size", "4", "--epochs", "4",
        "--score-microbatch", "1",
        "--eval-every", "1", "--eval-samples", "3", "--device", "cpu",
        "--tiny", "--no-ref", *extra])
    trainer.validate_args(args, dataset)
    return args


@pytest.fixture
def toy_backend(monkeypatch):
    model = TinyActor()
    monkeypatch.setattr(trainer, "build_models", lambda args, dev: (TinyTokenizer(), model, None))
    monkeypatch.setattr(trainer, "chat_ids", lambda tok, prompt, dev: torch.zeros((1, 1), dtype=torch.long))
    return model


@pytest.mark.parametrize("method,skip", [("ppo", False), ("grpo", True)])
def test_cpu_run_keeps_raw_answers_fixed_budgets_and_valid_done(tmp_path, dataset, toy_backend, method, skip):
    args = _args(tmp_path, dataset, method, ("--skip-heldout",) if skip else ())
    assert trainer.run(args) == 0
    out = tmp_path / "run"
    records = [json.loads(line) for line in (out / "completions.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    training = [row for row in records if row["phase"] == "train"]
    assert len(training) == 2 * 6 * 4
    assert all(row["completion_token_ids"] and isinstance(row["raw_text"], str) for row in records)
    assert all(row["scaled_reward"] == row["reward"] / 26 for row in records)
    heldout = [row for row in metrics if row["split"] == "heldout"]
    assert len(heldout) == (0 if skip else 1)
    if heldout:
        assert heldout[0]["step"] == 2 and heldout[0]["n_completions"] == 2 * 3
    updates = [row for row in metrics if row["phase"] == "train"]
    assert all(len(row["actor_epochs"]) == 4 for row in updates)
    assert all(row["actor_epochs"][0]["ratio_min"] == pytest.approx(1) for row in updates)
    assert all(row["actor_epochs"][0]["ratio_max"] == pytest.approx(1) for row in updates)
    assert toy_backend.forward_modes and not any(toy_backend.forward_modes)
    if method == "ppo":
        assert updates[0]["critic_mse_after"] < updates[0]["critic_mse_before"]
        assert any(row["old_value"] != 0 for row in training if row["step"] == 2)
        assert all(row["advantage"] == pytest.approx(row["scaled_reward"] - row["old_value"])
                   for row in training)
        assert (out / "critic.json").exists()
    else:
        assert all(row["critic_fit_steps"] == 0 for row in updates)
        for step in (1, 2):
            for instance in dataset["train"]:
                group = [row for row in training if row["step"] == step
                         and row["instance_id"] == instance["instance_id"]]
                assert sum(row["advantage"] for row in group) == pytest.approx(0, abs=1e-6)
    provenance = json.loads((out / "provenance.json").read_text())
    done = json.loads((out / "DONE").read_text())
    assert done["provenance_sha256"] == trainer._file_digest(out / "provenance.json")
    assert provenance["dataset_sha256"] == dataset["sha256"]
    assert provenance["source_sha256"] == trainer._digest(provenance["source_files"])
    assert all(trainer._file_digest(out / name) == digest
               for name, digest in provenance["artifact_sha256"].items())
    before = (out / "DONE").read_bytes()
    with pytest.raises(FileExistsError):
        trainer.run(args)
    assert (out / "DONE").read_bytes() == before


@pytest.mark.parametrize("group_size", [4, 8])
@pytest.mark.parametrize("microbatch", [1, 4])
def test_microbatch_gradients_match_full_batch_and_behavior_remains_frozen(tmp_path, dataset, group_size, microbatch):
    args = _args(tmp_path, dataset, extra=("--group-size", str(group_size)))
    args.score_microbatch = microbatch
    torch.manual_seed(23)
    model = TinyActor()
    critic = PromptValueCritic(6)
    prompts = [torch.zeros((1, 1), dtype=torch.long) for _ in dataset["train"]]
    groups = trainer.collect_rollout(model, TinyTokenizer(), prompts, dataset["train"], args, [], critic, 1)
    frozen = [group.batch.behavior_log_probs.clone() for group in groups]
    other = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    opt_other = torch.optim.SGD(other.parameters(), lr=0.5)
    micro = trainer.actor_update(model, opt, groups, args)
    args.score_microbatch = group_size
    full = trainer.actor_update(other, opt_other, groups, args)
    torch.testing.assert_close(model.logits, other.logits, atol=1e-7, rtol=1e-6)
    assert len(micro) == len(full) == 4
    assert any(abs(row["ratio_max"] - 1) > 1e-6 for row in micro[1:])
    for group, original in zip(groups, frozen):
        torch.testing.assert_close(group.batch.behavior_log_probs, original)


def test_all_critic_values_are_snapshotted_before_any_reward_is_observed(tmp_path, dataset, monkeypatch):
    args = _args(tmp_path, dataset)
    critic = PromptValueCritic(6)
    model = TinyActor()
    real_sample = trainer.sample_completions

    def sample_and_modify_critic(*positional):
        with torch.no_grad():
            critic.values.weight.fill_(7)
        return real_sample(*positional)

    monkeypatch.setattr(trainer, "sample_completions", sample_and_modify_critic)
    groups = trainer.collect_rollout(model, TinyTokenizer(),
        [torch.zeros((1, 1), dtype=torch.long) for _ in dataset["train"]],
        dataset["train"], args, [], critic, 1)
    assert all(torch.count_nonzero(group.batch.old_values) == 0 for group in groups)


def test_evaluation_preserves_training_rng_and_counts_invalid_outputs(tmp_path, dataset):
    args = _args(tmp_path, dataset)
    trainer.seed_everything(47)
    expected = torch.rand(5)
    trainer.seed_everything(47)
    handle = io.StringIO()
    result = trainer.evaluate(TinyActor(), TinyTokenizer(),
        [torch.zeros((1, 1), dtype=torch.long) for _ in dataset["train"]],
        dataset["train"], args, [], 0, "train", handle)
    torch.testing.assert_close(torch.rand(5), expected)
    records = [json.loads(line) for line in handle.getvalue().splitlines()]
    assert result["n_completions"] == 18
    assert result["useful_value_mean"] == pytest.approx(sum(row["useful_value"] for row in records) / 18)
    assert result["n_format_valid"] == sum(row["format_valid"] for row in records)
    assert all(row["useful_value"] == 0 for row in records if not row["useful_feasible"])


def test_failure_does_not_write_completion_marker(tmp_path, dataset, toy_backend, monkeypatch):
    args = _args(tmp_path, dataset)

    def invalid_update(*unused):
        return [{"policy_loss": float("nan")}]

    monkeypatch.setattr(trainer, "actor_update", invalid_update)
    with pytest.raises(ValueError, match="Out of range float"):
        trainer.run(args)
    assert not (tmp_path / "run" / "DONE").exists()


def test_reward_scale_must_bound_the_binary_targets(tmp_path, dataset):
    args = _args(tmp_path, dataset)
    args.reward_scale = 25
    with pytest.raises(ValueError, match="fixed reward scale"):
        trainer.validate_args(args, dataset)


def test_heldout_rewards_do_not_choose_the_training_scale(tmp_path, dataset):
    args = _args(tmp_path, dataset)
    altered = copy.deepcopy(dataset)
    altered["heldout"][0]["maximum_task_value"] = 1000
    trainer.validate_args(args, altered)
    assert args.reward_scale == 26


def test_generation_resets_saved_biases_and_masks_every_stop_token(tmp_path, dataset):
    from transformers import GenerationConfig
    args = _args(tmp_path, dataset)
    model = TinyActor()
    model.generation_config = GenerationConfig(eos_token_id=[2, 3], top_p=0.5,
        top_k=1, repetition_penalty=1.2, suppress_tokens=[1], forced_eos_token_id=2)
    eos_ids, recorded = trainer.configure_generation(model, TinyTokenizer(), args)
    assert eos_ids == [2, 3]
    assert recorded["top_p"] == recorded["repetition_penalty"] == recorded["temperature"] == 1
    assert recorded["top_k"] == 0
    assert recorded["suppress_tokens"] is None and recorded["forced_eos_token_id"] is None
    sequences = torch.tensor([[0, 1, 2, 0], [0, 3, 0, 0]])
    mask = trainer.completion_mask(sequences, 1, eos_ids)
    torch.testing.assert_close(mask.sum(1), torch.tensor([2, 1]))


def test_evaluation_draws_and_batches_do_not_change_with_training_group_size(tmp_path, dataset, monkeypatch):
    args = _args(tmp_path, dataset)
    args.eval_samples = 10
    model = TinyActor()
    prompts = [torch.zeros((1, 1), dtype=torch.long) for _ in dataset['train']]
    real_sample = trainer.sample_completions
    calls = []
    def tracked_sample(model, tok, prompt, args, count):
        calls.append(count)
        return real_sample(model, tok, prompt, args, count)
    monkeypatch.setattr(trainer, 'sample_completions', tracked_sample)
    outputs = []
    metrics = []
    for group_size in (4, 8):
        args.group_size = group_size
        handle = io.StringIO()
        metrics.append(trainer.evaluate(model, TinyTokenizer(), prompts, dataset['train'],
                                         args, [], 0, 'train', handle))
        outputs.append(handle.getvalue())
    assert calls == [4, 4, 2] * 12
    assert outputs[0] == outputs[1]
    assert metrics[0] == metrics[1]
