"""Condense a Kaggle/Colab run log or downloaded notebook into what matters.

Usage: python parse_run.py <file.log|file.txt|file.ipynb> [more files]

Kaggle logs come as "<seconds>s <line-number> <message>". Colab gives a .ipynb. Prints: the
environment block, job status counts, slow or failed jobs, tables (lines that look tabular),
verdict dicts, errors with tracebacks, and whether the run reached the end.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

NOISE = re.compile(
    r"Debugger warning|frozen modules|-Xfrozen_modules|PYDEVD_DISABLE|Debugging will proceed"
    r"|pip's dependency resolver|requires .* which is (not installed|incompatible)"
    r"|but you have .* which is incompatible|No files have been modified|No files matched"
    r"|Fetching \d+ files|it/s\]|s/it\]|Loading weights|Generating .* split|Tokenizing inputs"
    r"|Building contexts|^\s*$|LogBar|_pytree.py|register_constant|SyntaxWarning|re\.sub\("
    r"|text = re.sub|cells\[i\]\[c\]")
ENV_KEYS = ("platform", "gpu", "torch", "cuda", "transformers", "python", "code_version")
JOB = re.compile(r"\[\s*(done|skipped|failed)\]\s+(\w+)\s+(\w+)\s+(\w+)\s+items=(\d+)\s+([\d.]+)s")
KAGGLE = re.compile(r"^\s*([\d.]+)s\s+\d+\s?(.*)$")


def lines_of(path: Path) -> list[tuple[float | None, str]]:
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        out = []
        for cell in nb.get("cells", []):
            for o in cell.get("outputs", []):
                if o.get("output_type") == "stream":
                    text = "".join(o.get("text", ""))
                elif o.get("output_type") == "error":
                    text = "\n".join([f"{o.get('ename')}: {o.get('evalue')}",
                                      *o.get("traceback", [])])
                    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
                else:
                    text = "".join(o.get("data", {}).get("text/plain", ""))
                out += [(None, line) for line in text.splitlines()]
        return out
    out = []
    for raw in path.read_text(errors="replace").splitlines():
        raw = raw.replace("\t", " ")
        m = KAGGLE.match(raw)
        out.append((float(m.group(1)), m.group(2)) if m else (None, raw))
    return out


def summarize(path: Path) -> None:
    rows = lines_of(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    print(f"=== {path} ({path.stat().st_size} bytes, sha256 {digest})")
    times = [t for t, _ in rows if t is not None]
    if times:
        print(f"elapsed: {max(times) / 3600:.2f} h")
    env = {}
    for _, line in rows:
        for key in ENV_KEYS:
            m = re.match(rf"\s*{key}:\s*(.+)$", line)
            if m and key not in env:
                env[key] = m.group(1).strip()
    print("env:", json.dumps(env))
    secrets = sorted({line.strip() for _, line in rows if "secret " in line and "not available"
                      in line})
    for s in secrets:
        print("secret:", s)
    jobs, seen = [], set()
    for _, line in rows:
        m = JOB.search(line)
        if m and line not in seen:  # Kaggle sometimes logs a line twice
            seen.add(line)
            jobs.append(m.groups())
    if jobs:
        counts = Counter(j[0] for j in jobs)
        print("jobs:", dict(counts), "| by task:",
              dict(Counter((j[0], j[2]) for j in jobs)))
        done = [j for j in jobs if j[0] == "done"]
        if done:
            secs = sorted(float(j[5]) for j in done)
            print(f"done job seconds: median {secs[len(secs) // 2]:.0f}, max {secs[-1]:.0f}, "
                  f"total {sum(secs) / 3600:.2f} h")
        for j in jobs:
            if j[0] == "failed":
                print("FAILED job:", " ".join(j))
    print("--- signal lines (tables, verdicts, errors) ---")
    keep = re.compile(r"Traceback|Error|error:|FAIL|PASS|INCOMPLETE|verdict|\{'|K1|FR3|K2|K3"
                      r"|^\s*status|^\s*(lang|task|model)\s+\w+|calibration:|Secondary"
                      r"|stopping|^\s*(done|skipped|failed)\s+\d+\s*$")

    def tabular(line):  # a table row: 3+ fields, at least 2 numeric or boolean
        fields = line.split()
        numeric = sum(bool(re.fullmatch(r"-?\d+(\.\d+)?|True|False|NaN", f)) for f in fields)
        return len(fields) >= 3 and numeric >= 2 and len(line) < 200
    shown = 0
    for t, line in rows:
        if NOISE.search(line) or JOB.search(line) or line.lstrip().startswith(("INFO", "WARN")):
            continue
        if keep.search(line) or tabular(line):
            stamp = f"{t:>8.0f}s " if t is not None else ""
            print(stamp + line.rstrip())
            shown += 1
            if shown > 400:
                print("... truncated")
                break
    finished = any("NbConvertApp] Writing" in line and "__results__" in line or
                   "Converting notebook" in line for _, line in rows)
    print("reached end of notebook:", finished if path.suffix != ".ipynb" else "n/a (ipynb)")


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        sys.exit(__doc__)
    hashes = {}
    for p in paths:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h in hashes:
            print(f"=== {p} is identical to {hashes[h]}; skipped")
            continue
        hashes[h] = p
        summarize(p)
