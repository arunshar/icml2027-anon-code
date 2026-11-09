"""Check the strict revision protocol without changing the historical experiment."""
from __future__ import annotations

import copy
import itertools
import json
import random
from fractions import Fraction

import numpy as np
import pytest

import capacity_task as historical
import revision_task as R
import rewards as RW


@pytest.fixture(scope="module")
def manifest():
    return R.build_dataset()


@pytest.fixture
def tiny_instance():
    return {"instance_id": "toy", "values": [2, 5, 7], "weights": [1, 2, 4], "capacity": 3}


@pytest.mark.parametrize("text", [
    "[true]", "[false]", "[1.0]", "[1e0]", "[null]", "[NaN]", "[Infinity]",
    "[0, 0]", "[-1]", "[16]", '["1"]', "[[1]]", "[1,]", "[01]",
    "[1] explanation", "prefix [1]", "[1] [2]", "```json\n[1]\n```",
    "{}", "null", "1", "nothing", "[foo]",
])
def test_strict_parser_rejects_invalid_formats(text):
    assert R.parse_indices(text) is None


@pytest.mark.parametrize("text, expected", [
    ("[]", ()), (" [ ] \n", ()), ("[0]", (0,)), ("[15, 0, 3]", (0, 3, 15)),
])
def test_strict_parser_accepts_one_complete_integer_list(text, expected):
    assert R.parse_indices(text) == expected


def test_raw_text_survives_scoring_and_serialization(tiny_instance):
    text = " \n[1, 0]\t"
    score = R.score_completion(text, tiny_instance)
    assert score["raw_text"] == text
    assert json.loads(json.dumps(score))["raw_text"] == text
    assert score["parsed_indices"] == [0, 1]
    assert score["format_valid"] and score["nonempty"] and score["useful_feasible"]
    assert score["selected_weight"] == 3 and score["selected_value"] == 7
    assert score["useful_value"] == 1 and score["task_term"] == 1


def test_empty_output_is_parseable_but_never_useful(tiny_instance):
    for arm in R.ARMS:
        score = R.score_completion("[]", tiny_instance, arm, 5)
        assert score["format_valid"] and score["capacity_feasible"]
        assert not score["nonempty"] and not score["useful_feasible"]
        assert score["failure_reason"] == "empty_selection"
        assert score["task_term"] == R.INVALID_TASK
        assert score["useful_value"] == 0 and score["normalized_value"] == 0
        assert score["reward"] < 0
        assert score["reward"] == R.score_completion("[foo]", tiny_instance, arm, 5)["reward"]


def test_capacity_validity_and_usefulness_are_separate(tiny_instance):
    valid = R.score_completion("[0]", tiny_instance)
    overfull = R.score_completion("[2]", tiny_instance)
    invalid = R.score_completion("[3]", tiny_instance)
    assert valid["useful_feasible"] and valid["useful_value"] == 2 / 7
    assert overfull["format_valid"] and overfull["nonempty"]
    assert not overfull["capacity_feasible"] and overfull["useful_value"] == 0
    assert overfull["task_term"] == 1
    assert overfull["capacity_residual"] == pytest.approx(1 / 3)
    assert not invalid["format_valid"] and invalid["parse_error"] == "out_of_range_index"
    assert invalid["normalized_value"] is None and invalid["useful_value"] == 0


def test_binary_verdict_still_cancels_on_all_failing_groups(tiny_instance):
    outputs = ["[2]", "[0, 2]", "[]", "invalid text"]
    assert not any(R.score_completion(text, tiny_instance)["useful_feasible"] for text in outputs)
    task_only = R.group_rewards(outputs, tiny_instance, "none", 0)
    for weight in [0.5, 5.0, 25.0]:
        binary = R.group_rewards(outputs, tiny_instance, "binary", weight)
        assert np.allclose(binary, task_only - weight, rtol=0, atol=1e-14)
        assert np.allclose(RW.advantage(binary), RW.advantage(task_only), rtol=0, atol=1e-12)


def test_invalid_residual_exceeds_every_nonempty_overload(tiny_instance):
    invalid = R.score_completion("not JSON", tiny_instance, "graded", 5)
    all_items = R.score_completion("[0, 1, 2]", tiny_instance, "graded", 5)
    assert invalid["reward_residual"] == pytest.approx(all_items["capacity_residual"] + 1)
    assert invalid["reward"] < all_items["reward"]
    assert invalid["capacity_residual"] is None


def test_finite_reward_table_agrees_with_exhaustive_direct_scoring(tiny_instance):
    table = R.output_table(tiny_instance)
    assert len(table["task"]) == 2 ** len(tiny_instance["values"]) + 1
    for mask in range(2 ** len(tiny_instance["values"])):
        indices = [index for index in range(len(tiny_instance["values"])) if mask & (1 << index)]
        for arm in R.ARMS:
            score = R.score_completion(json.dumps(indices), tiny_instance, arm, 5)
            assert score["task_term"] == table["task"][mask]
            assert score["useful_feasible"] == table["verdict"][mask]
            assert score["reward_residual"] == table["residual"][mask]
            expected = table["task"][mask]
            if arm == "binary":
                expected -= 5 * (not table["verdict"][mask])
            elif arm == "graded":
                expected -= 5 * R.RESIDUAL_SCALE * table["residual"][mask]
            assert score["reward"] == expected
    invalid = R.score_completion("[true]", tiny_instance)
    assert table["subset_mask"][-1] == -1
    assert table["task"][-1] == invalid["task_term"]
    assert not table["format_valid"][-1] and not table["verdict"][-1]


def test_crossover_matches_independent_fraction_arithmetic(tiny_instance):
    feasible_values, failing_values = [], [-7]
    for k in range(4):
        for subset in itertools.combinations(range(3), k):
            value = sum(tiny_instance["values"][i] for i in subset)
            weight = sum(tiny_instance["weights"][i] for i in subset)
            if subset and weight <= tiny_instance["capacity"]:
                feasible_values.append(value)
            else:
                failing_values.append(value if subset else -7)
    expected = Fraction(max(failing_values) - min(feasible_values), 7)
    details = R.crossover_details(tiny_instance)
    assert Fraction(details["crossover_numerator"], details["crossover_denominator"]) == expected
    table = R.output_table(tiny_instance)
    for offset in [-1e-6, 0, 1e-6]:
        rewards = table["task"] - (float(expected) + offset) * ~table["verdict"]
        margin = rewards[table["verdict"]].min() - rewards[~table["verdict"]].max()
        assert margin == pytest.approx(offset, abs=1e-12)


def test_feasible_and_literal_anchors_cover_the_full_reward_space(tiny_instance):
    for arm in R.ARMS:
        for weight in [0, 5, 25]:
            rewards = [R.score_completion(json.dumps([i for i in range(3) if mask & (1 << i)]),
                                           tiny_instance, arm, weight)["reward"]
                       for mask in range(8)]
            rewards.append(R.score_completion("invalid", tiny_instance, arm, weight)["reward"])
            assert R.reward_anchor(tiny_instance, "ngrpo_feasible", arm, weight) == 1
            assert R.reward_anchor(tiny_instance, "ngrpo_sup", arm, weight) == max(rewards)


def test_scoring_uses_the_instance_instead_of_a_stored_optimum(tiny_instance, monkeypatch):
    tiny_instance["optimal_value"] = 999999
    monkeypatch.setattr(R, "output_table", lambda *args: pytest.fail("Scoring must not enumerate subsets."))
    assert R.score_completion("[0, 1]", tiny_instance)["useful_value"] == 1


def test_cached_output_arrays_cannot_be_mutated(tiny_instance):
    first, second = R.output_table(tiny_instance), R.output_table(tiny_instance)
    assert first["task"] is second["task"]
    with pytest.raises(ValueError):
        first["task"][0] = 999


def test_dataset_preserves_training_sets_and_keeps_splits_disjoint(manifest):
    assert len(manifest["train"]) == 6 and len(manifest["heldout"]) == 24
    for current, prior in zip(manifest["train"], historical.INSTANCES):
        assert current["values"] == prior["values"]
        assert current["weights"] == prior["weights"]
    identities = [(tuple(row["values"]), tuple(row["weights"]))
                  for row in manifest["train"] + manifest["heldout"]]
    assert len(set(identities)) == len(identities)
    assert manifest["generation"]["candidates_drawn"] == 24 + sum(manifest["generation"]["rejections"].values())


def test_every_prompt_states_its_capacity_and_nonempty_requirement(manifest):
    for instance in manifest["train"] + manifest["heldout"]:
        assert f"The capacity is {instance['capacity']}." in instance["prompt"]
        assert "at least one item" in instance["prompt"]
        assert instance["optimal_value"] > 0
        assert instance["maximum_task_value"] == sum(instance["values"]) / instance["optimal_value"]
        assert instance["enumerated_output_classes"] == (1 << 16) + 1
        table = R.output_table(instance)
        assert table["task_numerator"][table["verdict"]].max() == instance["optimal_value"]


def test_common_high_weight_ranks_every_feasible_subset_above_every_failure(manifest):
    high = manifest["common_above_crossover_weight"]
    exact = max(Fraction(row["crossover_numerator"], row["crossover_denominator"])
                for row in manifest["train"] + manifest["heldout"])
    assert manifest["global_crossover"] == float(exact)
    assert high > exact
    for instance in manifest["train"] + manifest["heldout"]:
        table = R.output_table(instance)
        rewards = table["task"] - high * ~table["verdict"]
        assert rewards[table["verdict"]].min() > rewards[~table["verdict"]].max()


def _same_numpy_state(left, right):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def test_generation_is_repeatable_and_does_not_change_global_random_states():
    py_state, np_state = random.getstate(), np.random.get_state()
    first = R.build_dataset(heldout_count=3, seed=8723)
    second = R.build_dataset(heldout_count=3, seed=8723)
    assert first == second
    assert random.getstate() == py_state
    assert _same_numpy_state(np.random.get_state(), np_state)
    changed = R.build_dataset(heldout_count=3, seed=8724)
    assert changed["sha256"] != first["sha256"]


def test_frozen_manifest_round_trips_without_using_a_random_generator(manifest, tmp_path, monkeypatch):
    target = tmp_path / "frozen.json"
    R.save_manifest(manifest, target)
    monkeypatch.setattr(np.random, "Generator", lambda *args: pytest.fail("Loading must not draw instances."))
    assert R.load_manifest(target) == manifest


def test_manifest_rejects_changed_content_and_rehashed_wrong_derivations(manifest, tmp_path):
    changed = copy.deepcopy(manifest)
    changed["heldout"][0]["values"][0] += 1
    with pytest.raises(ValueError, match="digest"):
        R.validate_manifest(changed)
    changed = copy.deepcopy(manifest)
    changed["heldout"][0]["optimal_value"] += 1
    changed["sha256"] = R.manifest_digest(changed)
    with pytest.raises(ValueError, match="instance"):
        R.validate_manifest(changed)
    changed = copy.deepcopy(manifest)
    changed["heldout"][0] = copy.deepcopy(changed["train"][0])
    changed["sha256"] = R.manifest_digest(changed)
    with pytest.raises(ValueError, match="duplicate"):
        R.validate_manifest(changed)


@pytest.mark.parametrize("change", ["short_instance", "out_of_range_value", "negative_rejection_count"])
def test_manifest_rejects_content_outside_the_declared_generation_protocol(manifest, change):
    changed = copy.deepcopy(manifest)
    if change == "short_instance":
        changed["heldout"][0]["values"].pop()
        changed["heldout"][0]["weights"].pop()
    elif change == "out_of_range_value":
        changed["heldout"][0]["values"][0] = 21
    else:
        changed["generation"]["rejections"]["duplicate_item_set"] = -1
    changed["sha256"] = R.manifest_digest(changed)
    with pytest.raises(ValueError):
        R.validate_manifest(changed)


@pytest.mark.parametrize("kwargs", [
    {"capacity_ratio": 0}, {"capacity_ratio": 1}, {"capacity_ratio": float("nan")},
    {"heldout_count": 0}, {"heldout_count": 2.5}, {"seed": -1}, {"seed": True},
])
def test_dataset_rejects_invalid_generation_settings(kwargs):
    with pytest.raises(ValueError):
        R.build_dataset(**kwargs)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf"), True])
def test_reward_rejects_invalid_penalty_weights(tiny_instance, weight):
    with pytest.raises(ValueError):
        R.score_completion("[0]", tiny_instance, weight=weight)
