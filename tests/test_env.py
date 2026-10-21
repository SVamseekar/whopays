import sys
import types

import pytest

from whopays import env

PLATFORM_VARS = ("COLAB_RELEASE_TAG", "COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (*PLATFORM_VARS, "WHOPAYS_RESULTS", "WHOPAYS_CODE_VERSION", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    ("var", "expected"),
    [("COLAB_RELEASE_TAG", "colab"), ("KAGGLE_KERNEL_RUN_TYPE", "kaggle"), (None, "local")],
)
def test_platform(monkeypatch, var, expected):
    if var:
        monkeypatch.setenv(var, "x")
    assert env.platform() == expected


@pytest.mark.parametrize(
    ("var", "root"),
    [
        ("COLAB_RELEASE_TAG", "/content/whopays-results"),
        ("KAGGLE_KERNEL_RUN_TYPE", "/kaggle/working/results"),
    ],
)
def test_results_root_per_platform(monkeypatch, var, root):
    monkeypatch.setenv(var, "x")
    assert str(env.results_root()) == root


def test_results_root_local_and_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert env.results_root() == tmp_path / "results"
    monkeypatch.setenv("WHOPAYS_RESULTS", "/somewhere")
    assert str(env.results_root()) == "/somewhere"


def test_secret_env_var_wins(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    monkeypatch.setenv("COLAB_RELEASE_TAG", "x")
    assert env.get_secret("HF_TOKEN") == "from-env"


def test_secret_colab(monkeypatch):
    monkeypatch.setenv("COLAB_RELEASE_TAG", "x")
    userdata = types.SimpleNamespace(get=lambda name: {"HF_TOKEN": "colab-tok"}[name])
    colab = types.ModuleType("google.colab")
    colab.userdata = userdata
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    assert env.get_secret("HF_TOKEN") == "colab-tok"
    assert env.get_secret("MISSING") is None  # KeyError from userdata is swallowed


def test_secret_kaggle(monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")

    class Client:
        def get_secret(self, name):
            return "kaggle-tok"

    mod = types.ModuleType("kaggle_secrets")
    mod.UserSecretsClient = Client
    monkeypatch.setitem(sys.modules, "kaggle_secrets", mod)
    assert env.get_secret("HF_TOKEN") == "kaggle-tok"


def test_secret_local_missing():
    assert env.get_secret("HF_TOKEN") is None


def test_device_is_valid():
    assert env.device() in {"cuda", "mps", "cpu"}


def test_code_version_override(monkeypatch):
    monkeypatch.setenv("WHOPAYS_CODE_VERSION", "abc123")
    assert env.code_version() == "abc123"


def test_env_info_has_required_columns():
    info = env.env_info()
    assert {"gpu", "torch", "transformers", "code_version"} <= info.keys()
