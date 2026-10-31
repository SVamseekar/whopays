"""Repair a broken optional vision import without replacing the platform torch build."""

import subprocess
import sys
from importlib import metadata
from types import SimpleNamespace
from unittest import mock

import pytest

from whopays import env


def test_absent_torchvision_needs_no_probe_or_install():
    with (mock.patch.object(env.metadata, 'version', side_effect=metadata.PackageNotFoundError),
          mock.patch.object(env.subprocess, 'run') as run):
        assert not env.prepare_text_runtime()
        run.assert_not_called()


def test_healthy_torchvision_is_preserved():
    with (mock.patch.object(env.metadata, 'version', return_value='test'),
          mock.patch.object(env.subprocess, 'run', return_value=SimpleNamespace(returncode=0))
          as run):
        assert not env.prepare_text_runtime()
        assert run.call_count == 1
        assert run.call_args.args[0] == [sys.executable, '-c', 'import torchvision']


@pytest.mark.parametrize('already_imported', [False, True])
def test_broken_optional_vision_is_removed_and_loaded_imports_require_restart(
        monkeypatch, already_imported):
    monkeypatch.delitem(sys.modules, 'torchvision', raising=False)
    monkeypatch.delitem(sys.modules, 'transformers', raising=False)
    if already_imported:
        monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace())
    with (mock.patch.object(env.metadata, 'version', return_value='test'),
          mock.patch.object(env.subprocess, 'run', side_effect=[
              SimpleNamespace(returncode=1, stderr='operator torchvision::nms does not exist'),
              SimpleNamespace(returncode=0),
          ]) as run):
        if already_imported:
            with pytest.raises(RuntimeError, match='Restart'):
                env.prepare_text_runtime()
        else:
            assert env.prepare_text_runtime()
        assert run.call_count == 2
        assert run.call_args.args[0] == [sys.executable, '-m', 'pip', 'uninstall', '-y',
                                        'torchvision']
        assert run.call_args.kwargs['check'] is True


def test_probe_timeout_does_not_remove_an_unverified_package():
    with (mock.patch.object(env.metadata, 'version', return_value='test'),
          mock.patch.object(env.subprocess, 'run',
                            side_effect=subprocess.TimeoutExpired('import torchvision', 60))
          as run):
        with pytest.raises(subprocess.TimeoutExpired):
            env.prepare_text_runtime()
        assert run.call_count == 1
