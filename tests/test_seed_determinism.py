import json

import train_speed_grpo as T


def _run(tmp_path, seed, name):
    out = tmp_path / name
    T.main(["--arm", "binary", "--w", "5", "--vmax", "30", "--steps", "2", "--tiny", "--device", "cpu",
            "--out-dir", str(out), "--eval-every", "0", "--batch-prompts", "2", "--seed", str(seed)])
    recs = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if '"train"' in l]
    speeds = [g["speeds"] for r in recs for g in r["groups"]]
    rewards = [g["rewards"] for r in recs for g in r["groups"]]
    prov = json.loads((out / "provenance.json").read_text())
    return speeds, rewards, prov["final_param_checksum"]


def test_same_seed_same_speeds_rewards_and_checksum(tmp_path):
    a = _run(tmp_path, 0, "a"); b = _run(tmp_path, 0, "b")
    assert a == b


def test_different_seed_changes_something(tmp_path):
    a = _run(tmp_path, 0, "a"); c = _run(tmp_path, 1, "c")
    assert a != c
