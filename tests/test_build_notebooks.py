import importlib.util
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bn", ROOT / "scripts" / "build_notebooks.py")
bn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bn)


def test_setup_cell_prepended_with_ref(tmp_path):
    src = tmp_path / "01_demo.py"
    src.write_text("# %%\nimport whopays\nwhopays.run('jobs.yaml')\n")
    nb = nbformat.reads(bn.build(src, ref="abc123"), as_version=4)
    nbformat.validate(nb)
    assert 'WHOPAYS_REF = "abc123"' in "".join(c.source for c in nb.cells)
    assert nb.cells[-1].source == "import whopays\nwhopays.run('jobs.yaml')"
    assert nb.metadata["colab"]["gpuType"] == "T4"


def test_standalone_has_no_setup_cell(tmp_path):
    src = tmp_path / "00_x.py"
    src.write_text("# whopays: standalone\n\n# %%\nprint(1)\n")
    nb = nbformat.reads(bn.build(src, ref="main"), as_version=4)
    assert [c.source for c in nb.cells] == ["print(1)"]


def test_build_is_deterministic(tmp_path):
    src = tmp_path / "01_demo.py"
    src.write_text("# %% [markdown]\n# Title\n\n# %%\nx = 1\n")
    assert bn.build(src, "main") == bn.build(src, "main")


def test_committed_notebooks_are_fresh():
    for path in sorted(bn.SRC.glob("[!_]*.py")):
        target = bn.OUT / f"{path.stem}.ipynb"
        assert target.read_text() == bn.build(path, "main"), f"rebuild {target.name}"


def test_notebook_sources_define_names_before_use():
    # Notebook cells never run in CI, so a cell reordering can only be caught statically.
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--isolated", "--select", "F821",
         *[str(p) for p in sorted(bn.SRC.glob("[!_]*.py"))]],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
