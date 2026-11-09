"""The decoding audit detects text and prompt corruption independently of rewards."""
import pytest
import audit_revision_tokens as audit


class ToyTokenizer:
    def get_vocab(self):
        return {"[": 1, "0": 2, "]": 3, "<stop>": 4, " ": 5}

    def batch_decode(self, rows, skip_special_tokens):
        assert skip_special_tokens
        return ["".join({1: "[", 2: "0", 3: "]", 4: "", 5: " "}[token]
                        for token in row) for row in rows]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return messages[0]["content"] + "answer"

    def __call__(self, text):
        return {"input_ids": [ord(character) for character in text]}


def test_completion_text_is_replayed_exactly():
    record = {"completion_token_ids": [1, 2, 3, 4], "n_completion_tokens": 4, "raw_text": "[0]"}
    assert audit.audit_completion_records(ToyTokenizer(), [record]) == 1
    with pytest.raises(ValueError, match="decoding"):
        audit.audit_completion_records(ToyTokenizer(), [{**record, "raw_text": "[1]"}])
    with pytest.raises(ValueError, match="decoding"):
        audit.audit_completion_records(ToyTokenizer(), [{**record, "raw_text": " [0]"}])


@pytest.mark.parametrize("tokens,count", [([], 0), ([True], 1), ([-1], 1), ([1.0], 1), ([1], 2)])
def test_invalid_token_sequences_are_rejected(tokens, count):
    with pytest.raises(ValueError, match="invalid"):
        audit.audit_completion_records(ToyTokenizer(), [{"completion_token_ids": tokens,
            "n_completion_tokens": count, "raw_text": ""}])


def test_saved_prompt_tokens_must_match_the_dataset_and_template():
    dataset = {"train": [{"instance_id": "train0", "prompt": "p"}], "heldout": []}
    saved = {"train": [{"instance_id": "train0", "prompt_token_ids": [ord(c) for c in "panswer"]}], "heldout": []}
    assert audit.audit_prompt_records(ToyTokenizer(), dataset, saved) == 1
    saved["train"][0]["prompt_token_ids"][0] += 1
    with pytest.raises(ValueError, match="prompt tokens"):
        audit.audit_prompt_records(ToyTokenizer(), dataset, saved)


def test_fast_tokenizer_cannot_silently_discard_an_out_of_domain_id():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]")),
        unk_token="[UNK]")
    ids = [1, 999999]
    assert tokenizer.decode(ids, skip_special_tokens=True) == "a"
    with pytest.raises(ValueError, match="outside the pinned tokenizer vocabulary"):
        audit.audit_completion_records(tokenizer, [{"completion_token_ids": ids,
            "n_completion_tokens": 2, "raw_text": "a"}])


def test_added_special_tokens_belong_to_the_explicit_domain():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]")),
        unk_token="[UNK]")
    tokenizer.add_special_tokens({"eos_token": "<stop>"})
    stop_id = tokenizer.eos_token_id
    assert stop_id >= tokenizer.vocab_size
    assert audit.audit_completion_records(tokenizer, [{"completion_token_ids": [1, stop_id],
        "n_completion_tokens": 2, "raw_text": "a"}]) == 1
    domain = audit.token_domain_record(tokenizer)
    assert domain["valid_id_count"] == len(tokenizer)
    assert "does not establish a training failure" in domain["unmapped_model_output_policy"]


def test_tokenizer_domain_is_an_exact_set_not_a_numeric_range():
    class SparseTokenizer(ToyTokenizer):
        def get_vocab(self):
            return {"[": 1, "]": 3, "<stop>": 4}

    with pytest.raises(ValueError, match="outside the pinned tokenizer vocabulary"):
        audit.audit_completion_records(SparseTokenizer(), [{"completion_token_ids": [1, 2, 3],
            "n_completion_tokens": 3, "raw_text": "[0]"}])


@pytest.mark.parametrize("vocabulary", [{}, {"bad": True}, {"bad": -1}, {"bad": 1.0}])
def test_malformed_explicit_vocabularies_are_rejected(vocabulary):
    class MalformedTokenizer(ToyTokenizer):
        def get_vocab(self):
            return vocabulary

    with pytest.raises(ValueError, match="valid explicit vocabulary"):
        audit.audit_completion_records(MalformedTokenizer(), [{"completion_token_ids": [1],
            "n_completion_tokens": 1, "raw_text": "["}])
