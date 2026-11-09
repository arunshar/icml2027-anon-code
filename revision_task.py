"""Define a strict capacity task without changing the historical experiment.

The revision protocol states the capacity in every prompt and evaluates nonempty
feasible selections on disjoint instances. Its training reward still separates the
varying task term from the binary verdict. Raw text remains part of every score record.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from capacity_task import INSTANCES as HISTORICAL_TRAIN_INSTANCES

PROTOCOL_VERSION = "strict_capacity_v1"
N_ITEMS = 16
DEFAULT_HELDOUT_SEED = 20260916
DEFAULT_HELDOUT_COUNT = 24
INVALID_TASK = -1.0
RESIDUAL_SCALE = 100.0
INVALID_RESIDUAL_MARGIN = 1.0
ARMS = ("none", "binary", "graded")


def _instance_key(instance: Mapping[str, Any]):
    values, weights = tuple(instance["values"]), tuple(instance["weights"])
    capacity = instance["capacity"]
    if not values or len(values) != len(weights) or len(values) > N_ITEMS:
        raise ValueError("An instance requires equally sized item arrays with at most sixteen items.")
    if any(type(x) is not int or x <= 0 for x in values + weights):
        raise ValueError("Every item value and weight must be a positive integer.")
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("The capacity must be a positive integer.")
    if min(weights) > capacity:
        raise ValueError("Every instance must admit a nonempty feasible selection.")
    if capacity >= sum(weights):
        raise ValueError("Every instance must also admit a nonempty violating selection.")
    return values, weights, capacity


@functools.lru_cache(maxsize=128)
def _dp_optimum(values, weights, capacity):
    best = [0] * (capacity + 1)
    for value, weight in zip(values, weights):
        for available in range(capacity, weight - 1, -1):
            best[available] = max(best[available], best[available - weight] + value)
    if best[-1] <= 0:
        raise ValueError("The instance has no positive nonempty optimum.")
    return best[-1]


def exact_optimum(instance):
    """Positive item values make the ordinary knapsack optimum nonempty here."""
    return _dp_optimum(*_instance_key(instance))


def _strict_parse(text: str, n_items: int):
    if not isinstance(text, str):
        return None, "completion_is_not_text"
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, ValueError):
        return None, "invalid_json"
    if type(value) is not list:
        return None, "not_a_list"
    if any(type(index) is not int for index in value):
        return None, "non_integer_index"
    if len(set(value)) != len(value):
        return None, "duplicate_index"
    if any(index < 0 or index >= n_items for index in value):
        return None, "out_of_range_index"
    return tuple(sorted(value)), None


def parse_indices(text: str, n_items: int = N_ITEMS):
    """Accept one complete JSON list while retaining an empty list as a distinct parse."""
    if type(n_items) is not int or n_items <= 0:
        raise ValueError("The item count must be a positive integer.")
    return _strict_parse(text, n_items)[0]


def _checked_weight(weight):
    if isinstance(weight, bool):
        raise ValueError("The reward weight must be a finite nonnegative number.")
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("The reward weight must be a finite nonnegative number.")
    return weight


def score_completion(text: str, instance, arm: str = "binary", weight: float = 5.0):
    """Return the raw completion, separate validity flags, task metrics, and training reward."""
    if arm not in ARMS:
        raise ValueError(f"The reward arm must be one of {ARMS}.")
    penalty_weight = _checked_weight(weight)
    values, weights, capacity = _instance_key(instance)
    optimum = _dp_optimum(values, weights, capacity)
    parsed, parse_error = _strict_parse(text, len(values))
    format_valid = parsed is not None
    nonempty = format_valid and bool(parsed)
    total_weight = sum(weights[index] for index in parsed) if format_valid else None
    total_value = sum(values[index] for index in parsed) if format_valid else None
    capacity_feasible = format_valid and total_weight <= capacity
    useful_feasible = bool(nonempty and capacity_feasible)
    normalized_value = total_value / optimum if format_valid else None
    task_term = normalized_value if nonempty else INVALID_TASK
    capacity_residual = max(total_weight / capacity - 1, 0.0) if format_valid else None
    invalid_residual = sum(weights) / capacity - 1 + INVALID_RESIDUAL_MARGIN
    reward_residual = capacity_residual if nonempty else invalid_residual
    if arm == "none":
        reward = task_term
    elif arm == "binary":
        reward = task_term - penalty_weight * (not useful_feasible)
    else:
        reward = task_term - penalty_weight * RESIDUAL_SCALE * reward_residual
    if not format_valid:
        failure_reason = "invalid_format"
    elif not nonempty:
        failure_reason = "empty_selection"
    elif not capacity_feasible:
        failure_reason = "capacity_exceeded"
    else:
        failure_reason = None
    return {
        "raw_text": text,
        "instance_id": instance.get("instance_id"),
        "parsed_indices": list(parsed) if format_valid else None,
        "parse_error": parse_error,
        "format_valid": format_valid,
        "nonempty": bool(nonempty),
        "capacity_feasible": bool(capacity_feasible),
        "useful_feasible": useful_feasible,
        "failure_reason": failure_reason,
        "selected_weight": total_weight,
        "selected_value": total_value,
        "capacity": capacity,
        "optimal_value": optimum,
        "normalized_value": normalized_value,
        "useful_value": normalized_value if useful_feasible else 0.0,
        "task_term": task_term,
        "capacity_residual": capacity_residual,
        "reward_residual": reward_residual,
        "reward": float(reward),
    }


def group_rewards(completions: Sequence[str], instance, arm="binary", weight=5.0):
    """Every reward depends on its own completion and the fixed instance."""
    return np.asarray([score_completion(text, instance, arm, weight)["reward"]
                       for text in completions], dtype=np.float64)


@functools.lru_cache(maxsize=64)
def _enumerated_table(values, weights, capacity):
    n_items = len(values)
    masks = np.arange(1 << n_items, dtype=np.int64)
    bits = ((masks[:, None] >> np.arange(n_items, dtype=np.int64)) & 1).astype(np.int8)
    selected_values = bits @ np.asarray(values, dtype=np.int64)
    selected_weights = bits @ np.asarray(weights, dtype=np.int64)
    optimum = _dp_optimum(values, weights, capacity)
    # The last row represents every malformed output, which shares one reward.
    task_numerators = np.append(selected_values, int(INVALID_TASK * optimum))
    task_numerators[0] = int(INVALID_TASK * optimum)
    task = task_numerators / optimum
    feasible = np.append((masks > 0) & (selected_weights <= capacity), False)
    invalid_residual = sum(weights) / capacity - 1 + INVALID_RESIDUAL_MARGIN
    residual = np.append(np.maximum(selected_weights / capacity - 1, 0.0), invalid_residual)
    residual[0] = invalid_residual
    table = {
        "subset_mask": np.append(masks, -1),
        "task_numerator": task_numerators,
        "task": task,
        "residual": residual,
        "verdict": feasible,
        "format_valid": np.append(np.ones(len(masks), dtype=bool), False),
        "nonempty": np.append(masks > 0, False),
    }
    for array in table.values():
        array.setflags(write=False)
    return table


def output_table(instance):
    """Enumerate every subset plus the malformed-output reward class."""
    return dict(_enumerated_table(*_instance_key(instance)))


def crossover_details(instance):
    """The exact threshold includes empty and malformed outputs among the failures."""
    table = output_table(instance)
    optimum = exact_optimum(instance)
    min_feasible = int(table["task_numerator"][table["verdict"]].min())
    max_failure = int(table["task_numerator"][~table["verdict"]].max())
    threshold = Fraction(max_failure - min_feasible, optimum)
    return {
        "crossover": float(threshold),
        "crossover_numerator": threshold.numerator,
        "crossover_denominator": threshold.denominator,
        "minimum_feasible_task_value": min_feasible / optimum,
        "maximum_failing_task_value": max_failure / optimum,
        "enumerated_output_classes": len(table["task"]),
    }


def crossover_weight(instance):
    """Strict feasible-over-failing reward ordering holds above this exact threshold."""
    return crossover_details(instance)["crossover"]


def reward_anchor(instance, variant="ngrpo_feasible", arm="binary", weight=5.0):
    """The feasible anchor is the optimum, and the literal anchor maximizes this reward."""
    if variant not in ("ngrpo_feasible", "ngrpo_sup"):
        raise ValueError("The anchor variant must be ngrpo_feasible or ngrpo_sup.")
    if arm not in ARMS:
        raise ValueError(f"The reward arm must be one of {ARMS}.")
    weight = _checked_weight(weight)
    table = output_table(instance)
    if variant == "ngrpo_feasible":
        return float(table["task"][table["verdict"]].max())
    if arm == "binary":
        rewards = table["task"] - weight * ~table["verdict"]
    elif arm == "graded":
        rewards = table["task"] - weight * RESIDUAL_SCALE * table["residual"]
    else:
        rewards = table["task"]
    return float(rewards.max())


def format_prompt(instance):
    """Every prompt states both the optimization objective and the complete output contract."""
    values, weights, capacity = _instance_key(instance)
    items = ", ".join(f"{index}:(value={value},weight={weight})"
                      for index, (value, weight) in enumerate(zip(values, weights)))
    return (f"Choose items to maximize their total value. The capacity is {capacity}. "
            f"The total weight must be at most {capacity}, and you must choose at least one item. "
            f"The items are {items}. Reply with exactly one JSON list of distinct integer indices "
            f"from 0 to {len(values) - 1}. Do not include explanations or any text outside the list.")


def _capacity(weights, ratio):
    return max(1, int(round(ratio * sum(weights))))


def _annotate_instance(values, weights, capacity, split, index):
    instance = {"instance_id": f"{split}_{index:03d}", "split": split,
                "values": list(values), "weights": list(weights), "capacity": int(capacity)}
    _instance_key(instance)
    instance["optimal_value"] = exact_optimum(instance)
    instance["maximum_task_value"] = float(output_table(instance)["task"].max())
    instance.update(crossover_details(instance))
    instance["prompt"] = format_prompt(instance)
    return instance


def _content_key(values, weights):
    return tuple(values), tuple(weights)


def manifest_digest(manifest):
    """Hash every manifest field except the field that stores the digest itself."""
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_dataset(capacity_ratio=0.03, heldout_count=DEFAULT_HELDOUT_COUNT,
                  seed=DEFAULT_HELDOUT_SEED):
    """Preserve the six historical item sets and generate a disjoint evaluation split locally."""
    if not math.isfinite(capacity_ratio) or not 0 < capacity_ratio < 1:
        raise ValueError("The capacity ratio must lie strictly between zero and one.")
    if type(heldout_count) is not int or heldout_count < 1:
        raise ValueError("The held-out split must contain at least one instance.")
    if type(seed) is not int or seed < 0:
        raise ValueError("The held-out seed must be a nonnegative integer.")
    train = [_annotate_instance(row["values"], row["weights"], _capacity(row["weights"], capacity_ratio),
                                "train", index)
             for index, row in enumerate(HISTORICAL_TRAIN_INSTANCES)]
    occupied = {_content_key(row["values"], row["weights"]) for row in train}
    rng = np.random.Generator(np.random.PCG64(seed))
    heldout = []
    rejections = {"duplicate_item_set": 0, "no_nonempty_feasible_subset": 0,
                  "no_violating_subset": 0}
    candidates = 0
    while len(heldout) < heldout_count:
        candidates += 1
        values = rng.integers(1, 21, size=N_ITEMS).tolist()
        weights = rng.integers(1, 21, size=N_ITEMS).tolist()
        key = _content_key(values, weights)
        capacity = _capacity(weights, capacity_ratio)
        if key in occupied:
            rejections["duplicate_item_set"] += 1
        elif min(weights) > capacity:
            rejections["no_nonempty_feasible_subset"] += 1
        elif sum(weights) <= capacity:
            rejections["no_violating_subset"] += 1
        else:
            occupied.add(key)
            heldout.append(_annotate_instance(values, weights, capacity, "heldout", len(heldout)))
    all_instances = train + heldout
    crossover = max(Fraction(row["crossover_numerator"], row["crossover_denominator"])
                    for row in all_instances)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "capacity_ratio": float(capacity_ratio),
        "n_items": N_ITEMS,
        "train_origin": "The six fixed item sets from the historical capacity protocol are reused.",
        "generation": {
            "heldout_seed": seed, "heldout_count": heldout_count,
            "generator": "numpy.random.Generator(PCG64)", "numpy_version": np.__version__,
            "candidate_order": "Each candidate draws all item values, then all item weights.",
            "value_range_inclusive": [1, 20], "weight_range_inclusive": [1, 20],
            "capacity_rule": "max(1, int(round(capacity_ratio * sum(weights))))",
            "rejection_rules": {
                "duplicate_item_set": "The ordered values and weights match an accepted training or held-out instance.",
                "no_nonempty_feasible_subset": "The lightest item exceeds the capacity.",
                "no_violating_subset": "All items together fit within the capacity.",
            },
            "candidates_drawn": candidates, "rejections": rejections,
        },
        "reward_protocol": {
            "invalid_or_empty_task_term": INVALID_TASK, "graded_residual_scale": RESIDUAL_SCALE,
            "invalid_residual_margin": INVALID_RESIDUAL_MARGIN,
            "invalid_residual_rule": "The reward residual exceeds the instance's largest possible overload by the fixed margin.",
            "binary_rule": "R = T - weight * I[not useful_feasible]",
            "graded_rule": "R = T - weight * graded_residual_scale * reward_residual",
            "useful_value_rule": "A valid nonempty feasible output receives V/OPT. Every other output receives zero.",
        },
        "train": train, "heldout": heldout,
        "global_crossover": float(crossover),
        "global_crossover_numerator": crossover.numerator,
        "global_crossover_denominator": crossover.denominator,
        "common_above_crossover_weight": crossover.numerator // crossover.denominator + 1,
    }
    manifest["sha256"] = manifest_digest(manifest)
    return manifest


def validate_manifest(manifest):
    """Verify the frozen content, its derived quantities, and split separation."""
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("The dataset protocol version is unsupported.")
    if manifest.get("sha256") != manifest_digest(manifest):
        raise ValueError("The manifest digest does not match its contents.")
    ratio = manifest.get("capacity_ratio")
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("The manifest's capacity ratio is invalid.")
    generation = manifest["generation"]
    if (manifest.get("n_items") != N_ITEMS
            or type(generation.get("heldout_seed")) is not int or generation["heldout_seed"] < 0
            or type(generation.get("heldout_count")) is not int or generation["heldout_count"] < 1):
        raise ValueError("The manifest's item count or generation settings are invalid.")
    rejections = generation.get("rejections", {})
    if (set(rejections) != {"duplicate_item_set", "no_nonempty_feasible_subset", "no_violating_subset"}
            or any(type(count) is not int or count < 0 for count in rejections.values())
            or generation.get("candidates_drawn") != generation["heldout_count"] + sum(rejections.values())):
        raise ValueError("The manifest's candidate and rejection counts are inconsistent.")
    if len(manifest["train"]) != len(HISTORICAL_TRAIN_INSTANCES):
        raise ValueError("The training split must preserve all six historical item sets.")
    if len(manifest["heldout"]) != manifest["generation"]["heldout_count"]:
        raise ValueError("The held-out split does not match its declared count.")
    occupied = set()
    for split in ("train", "heldout"):
        for index, instance in enumerate(manifest[split]):
            values, weights = instance["values"], instance["weights"]
            if (len(values) != N_ITEMS or len(weights) != N_ITEMS
                    or any(type(value) is not int or not 1 <= value <= 20
                           for value in values + weights)):
                raise ValueError("Every frozen instance must contain sixteen items in the declared integer ranges.")
            key = _content_key(instance["values"], instance["weights"])
            if key in occupied:
                raise ValueError("The dataset contains a duplicate item set across its splits.")
            occupied.add(key)
            if split == "train":
                original = HISTORICAL_TRAIN_INSTANCES[index]
                if key != _content_key(original["values"], original["weights"]):
                    raise ValueError("The training item sets differ from the historical instances.")
            expected = _annotate_instance(instance["values"], instance["weights"],
                                          _capacity(instance["weights"], manifest["capacity_ratio"]), split, index)
            if instance != expected:
                raise ValueError("An instance disagrees with its prompt, optimum, capacity, or crossover.")
    crossover = max(Fraction(row["crossover_numerator"], row["crossover_denominator"])
                    for row in manifest["train"] + manifest["heldout"])
    if (manifest["global_crossover"] != float(crossover)
            or Fraction(manifest["global_crossover_numerator"], manifest["global_crossover_denominator"]) != crossover
            or manifest["common_above_crossover_weight"] != crossover.numerator // crossover.denominator + 1):
        raise ValueError("The global crossover does not agree with the enumerated instances.")
    expected_reward = {"invalid_or_empty_task_term": INVALID_TASK,
                       "graded_residual_scale": RESIDUAL_SCALE,
                       "invalid_residual_margin": INVALID_RESIDUAL_MARGIN}
    if any(manifest["reward_protocol"].get(key) != value for key, value in expected_reward.items()):
        raise ValueError("The manifest's reward constants differ from this protocol.")
    return manifest


def save_manifest(manifest, path):
    """Write the frozen protocol before a training run begins."""
    validate_manifest(manifest)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return target


def load_manifest(path):
    """Read and validate a frozen dataset without drawing any random numbers."""
    return validate_manifest(json.loads(Path(path).read_text()))
