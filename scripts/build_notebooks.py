"""Build notebooks/*.ipynb from notebooks/src/*.py (jupytext percent format).

Every notebook gets the shared setup cell from notebooks/src/_setup.py prepended, unless its
source contains the line `# whopays: standalone`. Output is deterministic: notebook metadata
is fixed (Colab T4 runtime hint) and no jupytext version metadata is written.

    python scripts/build_notebooks.py               # build all
    python scripts/build_notebooks.py --check       # fail if any .ipynb is stale (CI)
    python scripts/build_notebooks.py --ref <sha>   # pin the commit notebooks install
    python scripts/build_notebooks.py --bundle      # also build dist/whopays-*.whl
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "src"
OUT = ROOT / "notebooks"
STANDALONE = "# whopays: standalone"
METADATA = {
    "accelerator": "GPU",
    "colab": {"gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def _cells(path: Path, ref: str | None = None) -> list:
    text = path.read_text()
    if ref is not None:
        text = text.replace("__WHOPAYS_REF__", ref)
    nb = jupytext.reads(text, fmt="py:percent")
    cells = []
    for cell in nb.cells:
        cell.metadata = {}
        cell.pop("id", None)
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
        cells.append(cell)
    return cells


def build(path: Path, ref: str) -> str:
    cells = [] if STANDALONE in path.read_text() else _cells(SRC / "_setup.py", ref)
    cells += [c for c in _cells(path) if c.source.strip() != STANDALONE]
    for i, cell in enumerate(cells):
        cell.id = f"c{i:03d}"  # stable IDs keep regenerated files byte-identical
    nb = nbformat.v4.new_notebook(cells=cells, metadata=METADATA)
    return nbformat.writes(nb) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="main", help="git ref the setup cell installs")
    parser.add_argument("--check", action="store_true", help="fail if outputs are stale")
    parser.add_argument("--bundle", action="store_true", help="also build a wheel into dist/")
    args = parser.parse_args()

    stale = []
    for path in sorted(SRC.glob("[!_]*.py")):
        target = OUT / f"{path.stem}.ipynb"
        text = build(path, args.ref)
        if args.check:
            if not target.exists() or target.read_text() != text:
                stale.append(target.name)
        else:
            target.write_text(text)
            print("built", target.relative_to(ROOT))
    if stale:
        print("stale notebooks (run scripts/build_notebooks.py):", ", ".join(stale))
        return 1
    if args.bundle:
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(ROOT / "dist"),
                        str(ROOT)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
