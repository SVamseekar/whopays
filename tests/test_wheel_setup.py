"""Exercise the generated offline setup without installing packages or using a GPU."""

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from whopays import env

ROOT = Path(__file__).resolve().parents[1]


def execute_setup(monkeypatch, tmp_path, *, include_lock=True, changed_loaded_build=False,
                  path_form='{path}', platform_env=(), namespace=None):
    wheel = tmp_path / 'whopays-0.1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        if include_lock:
            archive.writestr('whopays-0.1.0.data/data/share/whopays/requirements.lock',
                             'transformers==5.19.0\nnumpy==2.0.0\npandas==2.2.0\n')
    source = (ROOT / 'notebooks/src/_setup.py').read_text()
    value = path_form.format(path=wheel)
    source = source.replace('WHOPAYS_WHEEL = ""', f'WHOPAYS_WHEEL = {value!r}')
    source = source.replace('/tmp/whopays-constraints.txt', str(tmp_path / 'constraints.txt'))
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='Tesla T4\n')

    monkeypatch.delenv('COLAB_RELEASE_TAG', raising=False)
    monkeypatch.delenv('KAGGLE_KERNEL_RUN_TYPE', raising=False)
    monkeypatch.delenv('GH_TOKEN', raising=False)
    monkeypatch.delenv('HF_TOKEN', raising=False)
    for var in platform_env:
        monkeypatch.setenv(var, 'x')
    code = 'wheel:sha256:' + hashlib.sha256(wheel.read_bytes()).hexdigest()
    monkeypatch.setenv('WHOPAYS_CODE_VERSION', 'old-build' if changed_loaded_build else code)
    monkeypatch.setattr(subprocess, 'run', command)
    monkeypatch.setattr(env, 'prepare_text_runtime', lambda: False)
    monkeypatch.setattr(env, 'print_env', lambda: None)
    namespace = {} if namespace is None else namespace
    exec(compile(source, str(ROOT / 'notebooks/src/_setup.py'), 'exec'), namespace)
    return wheel, calls


def test_offline_wheel_install_uses_embedded_lock_and_content_identity(monkeypatch, tmp_path):
    wheel, calls = execute_setup(monkeypatch, tmp_path)
    pip = next(args for args in calls if 'install' in args)
    assert '-c' in pip
    constraints = Path(pip[pip.index('-c') + 1]).read_text()
    assert 'transformers==5.19.0' in constraints
    assert 'numpy==' not in constraints and 'pandas==' not in constraints
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert env.code_version() == f'wheel:sha256:{digest}'


def test_offline_wheel_without_lock_is_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match='requirements.lock'):
        execute_setup(monkeypatch, tmp_path, include_lock=False)


def test_same_version_wheel_is_reinstalled_without_forcing_dependency_reinstalls(
        monkeypatch, tmp_path):
    wheel, calls = execute_setup(monkeypatch, tmp_path)
    reinstalls = [args for args in calls if '--force-reinstall' in args]
    assert len(reinstalls) == 1
    assert '--no-deps' in reinstalls[0] and reinstalls[0][-1] == str(wheel)


def test_changed_build_requires_restart_if_package_is_already_imported(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match='[Rr]estart'):
        execute_setup(monkeypatch, tmp_path, changed_loaded_build=True)


@pytest.mark.parametrize('path_form', ['"{path}"', "'{path}'", '  "{path}"  '])
def test_colab_form_tolerates_pasted_quotes(monkeypatch, tmp_path, path_form):
    wheel, calls = execute_setup(monkeypatch, tmp_path, path_form=path_form)
    reinstall = next(args for args in calls if '--force-reinstall' in args)
    assert reinstall[-1] == str(wheel)


def test_kaggle_wins_over_colab_variables(monkeypatch, tmp_path):
    # Kaggle images are built on Colab's, so they also set COLAB_RELEASE_TAG; reading
    # Colab secrets there times out and leaves HF_TOKEN unset.
    namespace = {}
    monkeypatch.setitem(sys.modules, 'kaggle_secrets', None)
    execute_setup(monkeypatch, tmp_path, namespace=namespace,
                  platform_env=('COLAB_RELEASE_TAG', 'KAGGLE_KERNEL_RUN_TYPE'))
    assert namespace['PLATFORM'] == 'kaggle'
