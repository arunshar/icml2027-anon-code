import json
import os

import train_speed_grpo as T

REQUIRED = {"step", "phase", "s_per_step", "violation_rate", "parse_fail_rate", "all_violating_fraction",
            "mean_speed", "mean_residual", "reward_mean", "group_reward_std_mean", "violator_mean_adv",
            "frac_violators_pos_adv", "kl_seq_mean", "pg_loss", "groups"}


def test_three_steps_on_cpu_write_every_key_and_provenance(tmp_path):
    out = tmp_path / "run"
    rc = T.main(["--arm", "graded", "--w", "5", "--vmax", "30", "--steps", "3", "--tiny", "--device", "cpu",
                 "--out-dir", str(out), "--eval-every", "2", "--eval-samples", "2", "--batch-prompts", "2", "--seed", "0"])
    assert rc == 0
    recs = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in recs if r["phase"] == "train"]
    evals = [r for r in recs if r["phase"] == "eval"]
    assert [r["step"] for r in train] == [1, 2, 3]
    assert [r["step"] for r in evals] == [0, 2, 3]
    for r in train:
        assert REQUIRED <= set(r.keys())
        assert len(r["groups"]) == 2 and all(len(g["speeds"]) == 4 for g in r["groups"])
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["steps_completed"] == 3
    assert (out / "DONE").exists() and (out / "config.json").exists()
    assert prov["final_param_checksum"] and prov["git_sha"]
