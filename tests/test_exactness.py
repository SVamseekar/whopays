"""FR-5 exactness report plumbing; a planted divergence must fail the gate."""
import pandas as pd

from whopays.config import RunConfig
from whopays.decoding import exactness
from whopays.models.loader import load


def _bundle(path):
    cfg = RunConfig(model=str(path), task='mgsm', lang='en', spec=True, draft_model=str(path))
    return load(cfg, 'cpu'), cfg


def test_report_passes_on_exact_output_for_every_policy(tiny_model_dir):
    b, _ = _bundle(tiny_model_dir)
    prompts = {'en': [[1, 2, 3, 4], [5, 6, 7]], 'te': [[9, 8, 7, 6, 5]]}
    table = exactness.report(b, str(tiny_model_dir), prompts, max_new_tokens=12,
                             policies=('full', 'evict'), kv_budget=5, compressed_block=True)
    assert set(table.kv) == {'full', 'evict'}
    assert table.exact.sum() == table.n_prompts.sum() == 6
    assert exactness.verdict(table, min_prompts=3) == 'PASS'
    assert (table.spec_target_forwards <= table.plain_target_forwards).all()


def test_gate_runs_past_eos_so_flush_boundaries_are_exercised(tiny_model_dir, monkeypatch):
    b, _ = _bundle(tiny_model_dir)
    calls = []
    original = exactness.generate

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(exactness, 'generate', spy)
    exactness.report(b, str(tiny_model_dir), {'en': [[1, 2, 3]]}, max_new_tokens=4,
                     policies=('full',))
    assert all(call['eos_token_id'] == [] for call in calls)


def test_divergence_fails_the_gate():
    table = pd.DataFrame({'kv': ['evict'], 'lang': ['te'], 'n_prompts': [100], 'exact': [99],
                          'first_divergence': [17], 'plain_target_forwards': [10],
                          'spec_target_forwards': [5]})
    assert exactness.verdict(table, min_prompts=100) == 'FAIL'


def test_too_few_prompts_cannot_pass():
    table = pd.DataFrame({'kv': ['evict'], 'lang': ['te'], 'n_prompts': [5], 'exact': [5],
                          'first_divergence': [None], 'plain_target_forwards': [10],
                          'spec_target_forwards': [5]})
    assert exactness.verdict(table, min_prompts=100) == 'INCOMPLETE'


def test_every_language_must_be_covered():
    table = pd.DataFrame({'kv': ['full'], 'lang': ['en'], 'n_prompts': [100], 'exact': [100],
                          'first_divergence': [None], 'plain_target_forwards': [10],
                          'spec_target_forwards': [5]})
    assert exactness.verdict(table, min_prompts=100, languages=['en', 'te']) == 'INCOMPLETE'


def test_gate_prompts_match_the_task_encoding(monkeypatch):
    from whopays.tasks.mgsm import build_prompt

    monkeypatch.setattr('whopays.datasets.mgsm',
                        lambda lang: [{'question': f'q{i}', 'item_id': str(i)} for i in range(30)])

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is True
            return [ord(c) for c in text]

    prompts = exactness.mgsm_prompts(Tokenizer(), per_language=2, languages=['te'])
    assert len(prompts['te']) == 2
    assert prompts['te'][1] == [ord(c) for c in build_prompt('te', 'q1')]


def test_gate_reuses_a_stored_pass_for_the_same_code(tiny_model_dir, tmp_path, monkeypatch):
    # A session that dies during the speed grid must not repeat a finished gate.
    b, _ = _bundle(tiny_model_dir)
    monkeypatch.setattr(exactness, 'mgsm_prompts',
                        lambda tok, n: {lang: [[1, 2, 3]] * 10 for lang in exactness.CODES})
    monkeypatch.setattr(exactness, 'POLICIES', ('full',))
    calls = []
    original = exactness.report

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, max_new_tokens=2, policies=('full',))

    monkeypatch.setattr(exactness, 'report', counting)
    args = dict(root=tmp_path, sync=False, loader=lambda cfg: b)
    first, state = exactness.run_gate(str(tiny_model_dir), **args)
    assert state == 'PASS'
    second, state = exactness.run_gate(str(tiny_model_dir), **args)
    assert state == 'PASS' and len(calls) == 1 and len(second) == len(first)
