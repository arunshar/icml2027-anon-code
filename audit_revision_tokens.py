#!/usr/bin/env python3
"""Replay prompt tokenization and completion decoding for strict training records.

The caller must name an immutable tokenizer revision. This CPU audit does not load
model weights or change records. A decoding match supports the saved text and token
relationship. It does not establish an unrecorded historical tokenizer identity.
The audit accepts only IDs present in the pinned tokenizer's explicit vocabulary.
Some models have additional output rows with no corresponding tokenizer entry.
This audit cannot certify those model-reserved IDs and never invents text for them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_token_ids(tokenizer):
    """Return the exact tokenizer domain, including its added and special tokens."""
    vocabulary = tokenizer.get_vocab()
    if (not isinstance(vocabulary, dict) or not vocabulary
            or any(type(token) is not int or token < 0 for token in vocabulary.values())):
        raise ValueError("The tokenizer does not expose a valid explicit vocabulary.")
    return frozenset(vocabulary.values())


def token_domain_record(tokenizer):
    """Record the domain that the audit can certify without loading model weights."""
    domain = valid_token_ids(tokenizer)
    encoded = json.dumps(sorted(domain), separators=(",", ":")).encode()
    return {
        "definition": "The exact IDs returned by tokenizer.get_vocab(), including added tokens.",
        "valid_id_count": len(domain),
        "valid_ids_sha256": hashlib.sha256(encoded).hexdigest(),
        "unmapped_model_output_policy": (
            "The audit rejects IDs absent from the tokenizer vocabulary, including model-reserved output IDs. "
            "Such a rejection means this text audit cannot certify the sequence. It does not establish a training failure."),
    }


def audit_prompt_records(tokenizer, dataset, prompt_tokens):
    count = 0
    for split in ("train", "heldout"):
        instances, saved = dataset[split], prompt_tokens[split]
        if len(instances) != len(saved):
            raise ValueError("The saved prompt count differs from the dataset.")
        for instance, record in zip(instances, saved):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": instance["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            ids = tokenizer(rendered)["input_ids"]
            if (record["instance_id"] != instance["instance_id"]
                    or ids != record["prompt_token_ids"]):
                raise ValueError("Saved prompt tokens disagree with the declared tokenizer.")
            count += 1
    return count


def audit_completion_records(tokenizer, records):
    if not records:
        raise ValueError("The run has no completion records to decode.")
    domain = valid_token_ids(tokenizer)
    for offset in range(0, len(records), 512):
        batch = records[offset:offset + 512]
        ids = []
        for row in batch:
            tokens = row["completion_token_ids"]
            if (not isinstance(tokens, list) or not tokens
                    or any(type(token) is not int or token < 0 for token in tokens)
                    or type(row["n_completion_tokens"]) is not int
                    or len(tokens) != row["n_completion_tokens"]):
                raise ValueError("A completion has an invalid saved token sequence.")
            if any(token not in domain for token in tokens):
                raise ValueError(
                    "A saved token ID is outside the pinned tokenizer vocabulary. "
                    "This audit cannot certify model-reserved or unknown IDs as text tokens.")
            ids.append(tokens)
        decoded = tokenizer.batch_decode(ids, skip_special_tokens=True)
        for row, text in zip(batch, decoded):
            if text != row["raw_text"]:
                raise ValueError("Completion decoding disagrees with its saved raw text.")
    return len(records)


def audit_run(path, tokenizer, revision):
    path = Path(path)
    required = ("config.json", "dataset.json", "prompt_tokens.json", "provenance.json",
                "completions.jsonl", "DONE")
    before = {name: file_hash(path / name) for name in required}
    load = lambda name: json.loads((path / name).read_text())
    config, provenance, done = load("config.json"), load("provenance.json"), load("DONE")
    if (config.get("protocol_version") != "strict_capacity_training_v1"
            or done.get("status") != "complete"
            or done.get("provenance_sha256") != before["provenance.json"]):
        raise ValueError("The run is not a completed strict-protocol record.")
    recorded_revision = provenance.get("tokenizer_revision")
    if recorded_revision is not None and recorded_revision != revision:
        raise ValueError("The audit tokenizer revision differs from the recorded tokenizer revision.")
    if config["model_id"] != tokenizer.name_or_path:
        raise ValueError("The audit tokenizer does not match the run's declared model ID.")
    for name in ("config.json", "dataset.json", "prompt_tokens.json", "completions.jsonl"):
        if provenance["artifact_sha256"][name] != before[name]:
            raise ValueError("A run artifact differs from its recorded checksum.")
    records = [json.loads(line) for line in (path / "completions.jsonl").read_text().splitlines()]
    prompts = audit_prompt_records(tokenizer, load("dataset.json"), load("prompt_tokens.json"))
    completions = audit_completion_records(tokenizer, records)
    if {name: file_hash(path / name) for name in required} != before:
        raise ValueError("The run changed during token decoding verification.")
    return {"run": path.name, "completed_prompt_records": prompts,
            "completed_completion_records": completions, "input_sha256": before,
            "recorded_tokenizer_revision": recorded_revision,
            "audited_tokenizer_revision": revision,
            "scope": "Every saved prompt token sequence and completion text matches this tokenizer."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.tokenizer_revision):
        raise ValueError("The tokenizer revision must be an immutable 40-character commit digest.")
    if args.out.exists():
        raise ValueError("The output audit record already exists.")
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    model_ids = {json.loads((path / "config.json").read_text())["model_id"] for path in args.runs}
    if len(model_ids) != 1:
        raise ValueError("The requested runs must use one declared model ID.")
    model_id = next(iter(model_ids))
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=args.tokenizer_revision,
                                              local_files_only=True)
    tokenizer_files = {name: file_hash(hf_hub_download(model_id, name,
        revision=args.tokenizer_revision, local_files_only=True))
        for name in ("tokenizer.json", "tokenizer_config.json")}
    report = {"status": "token_text_replay_passed", "model_id": model_id,
              "tokenizer_revision": args.tokenizer_revision,
              "tokenizer_file_sha256": tokenizer_files,
              "token_domain": token_domain_record(tokenizer),
              "audit_source_sha256": file_hash(__file__),
              "runs": [audit_run(path, tokenizer, args.tokenizer_revision) for path in args.runs]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]),
                      "completions": sum(row["completed_completion_records"] for row in report["runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
