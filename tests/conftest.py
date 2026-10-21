"""Shared fixtures: a tiny random Qwen2 model with an offline byte-level BPE tokenizer.

Nothing is downloaded, so model tests run in CI on CPU in seconds. Outputs are meaningless;
tests check plumbing (shapes, caching, resume, scoring paths), never accuracy.
"""

import os
from pathlib import Path

import pytest

CORPUS = [
    "P: The river town kept its records in a stone archive by the bridge.\n",
    "Q: Where were the records kept?\nA: bridge\nB: archive\nC: river\nD: town\nAnswer: B",
    " A B C D Answer: A Answer: B Answer: C Answer: D 12 345 6789 = + - x",
    "Das Wetter ist schön. Сегодня хорошая погода. 今天天气很好。ఈ రోజు వాతావరణం బాగుంది.",
]


def build_tiny_model(path: Path) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS * 50, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>",
                                   pad_token="<|endoftext|>")
    fast.save_pretrained(path)

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
        eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id,
    )
    Qwen2ForCausalLM(config).save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    return build_tiny_model(tmp_path_factory.mktemp("tiny-qwen2"))


def pytest_collection_modifyitems(config, items):
    """Tests marked `slow` download real data or models; run them with WHOPAYS_SLOW=1."""
    if os.environ.get("WHOPAYS_SLOW"):
        return
    skip = pytest.mark.skip(reason="downloads data; set WHOPAYS_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
