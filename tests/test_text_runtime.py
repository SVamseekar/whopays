"""Repair broken optional media imports without replacing the platform torch build."""

import subprocess
import sys
from importlib import metadata
from types import SimpleNamespace
from unittest import mock

import pytest

from whopays import env


def installed(*packages):
    def version(name):
        if name not in packages:
            raise metadata.PackageNotFoundError(name)
        return 'test'
    return version


def test_absent_optional_packages_need_no_probe_or_install():
    with (mock.patch.object(env.metadata, 'version', side_effect=metadata.PackageNotFoundError),
          mock.patch.object(env.subprocess, 'run') as run):
        assert not env.prepare_text_runtime()
        run.assert_not_called()


@pytest.mark.parametrize('package', ['torchvision', 'torchaudio'])
def test_healthy_optional_package_is_preserved(package):
    with (mock.patch.object(env.metadata, 'version', side_effect=installed(package)),
          mock.patch.object(env.subprocess, 'run', return_value=SimpleNamespace(returncode=0))
          as run):
        assert not env.prepare_text_runtime()
        assert run.call_count == 1
        assert run.call_args.args[0] == [sys.executable, '-c', f'import {package}']


@pytest.mark.parametrize('already_imported', [False, True])
@pytest.mark.parametrize('package', ['torchvision', 'torchaudio'])
def test_broken_optional_package_is_removed_and_loaded_imports_require_restart(
        monkeypatch, already_imported, package):
    monkeypatch.delitem(sys.modules, 'torchvision', raising=False)
    monkeypatch.delitem(sys.modules, 'torchaudio', raising=False)
    monkeypatch.delitem(sys.modules, 'transformers', raising=False)
    if already_imported:
        monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace())
    with (mock.patch.object(env.metadata, 'version', side_effect=installed(package)),
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
                                        package]
        assert run.call_args.kwargs['check'] is True


def test_probe_timeout_does_not_remove_an_unverified_package():
    with (mock.patch.object(env.metadata, 'version', return_value='test'),
          mock.patch.object(env.subprocess, 'run',
                            side_effect=subprocess.TimeoutExpired('import torchvision', 60))
          as run):
        with pytest.raises(subprocess.TimeoutExpired):
            env.prepare_text_runtime()
        assert run.call_count == 1


def test_both_failures_are_repaired_together_before_requesting_restart(monkeypatch):
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace())
    with (mock.patch.object(env.metadata, 'version', return_value='test'),
          mock.patch.object(env.subprocess, 'run', side_effect=[
              SimpleNamespace(returncode=1), SimpleNamespace(returncode=1),
              SimpleNamespace(returncode=0),
          ]) as run):
        with pytest.raises(RuntimeError, match='Restart'):
            env.prepare_text_runtime()
        assert run.call_count == 3
        assert run.call_args.args[0] == [sys.executable, '-m', 'pip', 'uninstall', '-y',
                                        'torchvision', 'torchaudio']


@pytest.mark.parametrize('broken', ['torchvision', 'torchaudio'])
def test_healthy_companion_is_preserved(monkeypatch, broken):
    for name in ['transformers', 'torchvision', 'torchaudio']:
        monkeypatch.delitem(sys.modules, name, raising=False)

    def command(args, **kwargs):
        if '-c' in args:
            return SimpleNamespace(returncode=int(args[-1] == f'import {broken}'))
        assert args == [sys.executable, '-m', 'pip', 'uninstall', '-y', broken]
        return SimpleNamespace(returncode=0)

    with (mock.patch.object(env.metadata, 'version', return_value='test'),
          mock.patch.object(env.subprocess, 'run', side_effect=command)):
        assert env.prepare_text_runtime()
