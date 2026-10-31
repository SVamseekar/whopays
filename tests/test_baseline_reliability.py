"""CPU checks for the GPU baseline handoff; no benchmark measurements."""

import sqlite3
import subprocess
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from whopays import crosscheck, throughput
from whopays.grids import K1_MODEL


@pytest.mark.parametrize(('ratio', 'replan'), [(2.0, False), (2.004, True)])
def test_throughput_gate_does_not_round_before_deciding(ratio, replan):
    seconds = ratio * 15 * 60 / 9900
    rows = pd.DataFrame([dict(model=K1_MODEL, task='belebele', item_id='a',
                              wall_time_s=seconds)])
    result = throughput.summary(rows).iloc[0]
    assert result.ratio_vs_assumed == 2.0  # display rounds both cases identically
    assert bool(result.replan) is replan


def test_checkpoint_uploads_a_consistent_sqlite_snapshot(tmp_path):
    run_dir = tmp_path / 'lmeval-test'
    run_dir.mkdir()
    db = run_dir / 'cache_rank0.db'
    # Leave committed requests in a live WAL: copying only the main file loses these.
    with sqlite3.connect(db) as writer:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute('CREATE TABLE requests (answer TEXT)')
        writer.execute("INSERT INTO requests VALUES ('cached')")
        writer.commit()
        (run_dir / 'lmeval.log').write_text('progress\n')

        def check_snapshot(snapshot, **kwargs):
            assert snapshot.name == run_dir.name and snapshot != run_dir
            assert (snapshot / 'lmeval.log').read_text() == 'progress\n'
            with sqlite3.connect(snapshot / db.name) as reader:
                assert reader.execute('SELECT * FROM requests').fetchall() == [('cached',)]
                assert reader.execute('PRAGMA integrity_check').fetchone() == ('ok',)
            assert not list(snapshot.glob('*-wal'))
            return True

        with (mock.patch('whopays.env.get_secret', return_value='test-token'),
              mock.patch('whopays.io.hub_sync.push_run', side_effect=check_snapshot) as push):
            assert crosscheck.sync_lmeval(run_dir)
        assert push.call_count == 1


@pytest.mark.parametrize('returncode', [0, 1])
def test_reference_run_checkpoints_while_running_and_on_exit(tmp_path, returncode):
    proc = mock.MagicMock()
    proc.__enter__.return_value = proc
    proc.wait.side_effect = [subprocess.TimeoutExpired('lm_eval', 600), returncode]
    with (mock.patch.object(crosscheck, 'install', return_value=Path('/fake/python')),
          mock.patch.object(crosscheck.subprocess, 'Popen', return_value=proc),
          mock.patch.object(crosscheck, 'sync_lmeval', return_value=True) as sync):
        if returncode:
            with pytest.raises(RuntimeError, match='lm-eval failed'):
                crosscheck.run_lmeval(K1_MODEL, ['mgsm'], ['en'], tmp_path)
        else:
            crosscheck.run_lmeval(K1_MODEL, ['mgsm'], ['en'], tmp_path)
        assert sync.call_count == 2


def test_sync_failure_keeps_local_reference_results(tmp_path):
    with (mock.patch('whopays.env.get_secret', return_value='test-token'),
          mock.patch('whopays.io.hub_sync.push_run', side_effect=OSError('offline'))):
        assert not crosscheck.sync_lmeval(tmp_path)


def test_reference_run_without_sync_does_not_upload(tmp_path):
    proc = mock.MagicMock()
    proc.__enter__.return_value = proc
    proc.wait.return_value = 0
    with (mock.patch.object(crosscheck, 'install', return_value=Path('/fake/python')),
          mock.patch.object(crosscheck.subprocess, 'Popen', return_value=proc),
          mock.patch.object(crosscheck, 'sync_lmeval') as sync):
        crosscheck.run_lmeval(K1_MODEL, ['mgsm'], ['en'], tmp_path, sync=False)
        sync.assert_not_called()


def test_missing_token_skips_checkpoint_without_touching_files(tmp_path):
    with (mock.patch('whopays.env.get_secret', return_value=None),
          mock.patch('whopays.io.hub_sync.push_run') as push):
        assert not crosscheck.sync_lmeval(tmp_path / 'missing')
        push.assert_not_called()


def test_interrupt_stops_reference_process_before_final_checkpoint(tmp_path):
    proc = mock.MagicMock()
    proc.__enter__.return_value = proc
    proc.wait.side_effect = [KeyboardInterrupt, 0]
    with (mock.patch.object(crosscheck, 'install', return_value=Path('/fake/python')),
          mock.patch.object(crosscheck.subprocess, 'Popen', return_value=proc),
          mock.patch.object(crosscheck, 'sync_lmeval') as sync):
        with pytest.raises(KeyboardInterrupt):
            crosscheck.run_lmeval(K1_MODEL, ['mgsm'], ['en'], tmp_path)
        proc.terminate.assert_called_once()
        sync.assert_called_once_with(tmp_path)
