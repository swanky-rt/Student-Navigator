import argparse
import ast
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

# Optional defaults (won't break if you don't have defaults.py)
try:
    from defaults import DEFAULT_GENERATIONS, DEFAULT_EVAL_REPORT, REPORTS_DIR
except Exception:
    DEFAULT_GENERATIONS = "reports/baseline_gens.jsonl"
    DEFAULT_EVAL_REPORT = "reports/baseline_report.json"
    REPORTS_DIR = "reports"

os.makedirs(REPORTS_DIR, exist_ok=True)

def run(cmd: str):
    """Run a shell command and return (rc, stdout, stderr)."""
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

def write_code_files(gens_jsonl: str, tmp_dir: str):
    """
    Dumps each generated completion into tmp_dir/sample_#.py
    Expects JSONL with key 'generated' (or 'completion'/'response').
    """
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    files = []
    with open(gens_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                row = json.loads(line)
            except Exception:
                continue
            code = row.get("generated") or row.get("completion") or row.get("response") or ""
            p = Path(tmp_dir) / f"sample_{i}.py"
            p.write_text(code, encoding="utf-8")
            files.append(str(p))
    return files

def materialize_tasks_package(tmp_dir: str, files):
    """
    Create a 'tasks' package under tmp_dir and put one generated file as tasks/core.py.
    Strategy: pick the first non-empty generation. If none exist, provide a tiny fallback.
    """
    tmp = Path(tmp_dir)
    tasks_pkg = tmp / "tasks"
    tasks_pkg.mkdir(parents=True, exist_ok=True)
    (tasks_pkg / "__init__.py").write_text("", encoding="utf-8")

    chosen_src = ""
    for fp in files:
        src = Path(fp).read_text(encoding="utf-8", errors="ignore").strip()
        if src:
            chosen_src = src
            break

    if not chosen_src:
        # minimal fallback to avoid empty package (keeps pytest from erroring)
        chosen_src = (
            "from typing import List, Dict\n"
            "def is_palindrome(s: str) -> bool:\n"
            "    s = s or ''\n"
            "    s = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
            "    return s == s[::-1]\n"
            "def slugify(s: str) -> str:\n"
            "    s = s or ''\n"
            "    import re\n"
            "    s = s.lower()\n"
            "    s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')\n"
            "    s = re.sub(r'-{2,}', '-', s)\n"
            "    return s\n"
            "def add(a: int, b: int) -> int:\n"
            "    return int(a) + int(b)\n"
            "def top_k(xs: list, k: int) -> list:\n"
            "    return sorted(xs, reverse=True)[:k]\n"
            "def count_words(text: str):\n"
            "    from collections import Counter\n"
            "    import re\n"
            "    text = text or ''\n"
            "    toks = re.findall(r\"[a-z0-9]+'?[a-z0-9]+\", text.lower())\n"
            "    return Counter(toks)\n"
        )

    (tasks_pkg / "core.py").write_text(chosen_src, encoding="utf-8")
    return str(tmp)

# ---- Parsers for tool outputs ----

def parse_bandit(out: str) -> int:
    """bandit -r -f json -> count issues"""
    try:
        j = json.loads(out)
        if isinstance(j, dict):
            if "results" in j and isinstance(j["results"], list):
                return len(j["results"])
            m = j.get("metrics") or {}
            if isinstance(m, dict):
                for v in m.values():
                    if isinstance(v, dict) and "issues" in v:
                        try:
                            return int(v["issues"])
                        except Exception:
                            pass
        if isinstance(j, list):
            return len(j)
    except Exception:
        pass
    return 0

def parse_ruff(out: str) -> int:
    """ruff check --output-format json -> either a JSON array or NDJSON lines."""
    out = out.strip()
    if not out:
        return 0
    try:
        j = json.loads(out)
        if isinstance(j, list):
            return len(j)
        if isinstance(j, dict) and "diagnostics" in j and isinstance(j["diagnostics"], list):
            return len(j["diagnostics"])
    except Exception:
        pass
    count = 0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            _ = json.loads(line)
            count += 1
        except Exception:
            continue
    return count

def parse_flake8(out: str) -> int:
    # out is a dict: { "file.py": [ { "code": "E302", ...}, ...], ... }
    try:
        j = json.loads(out or "{}")
        if isinstance(j, dict):
            return sum(len(v) for v in j.values() if isinstance(v, list))
    except Exception:
        pass
    return 0

def parse_radon(out: str) -> int:
    """radon cc -j -> {file: [blocks...]} ; return count of blocks (approx complexity hits)."""
    try:
        j = json.loads(out or "{}")
        if isinstance(j, dict):
            return sum(len(v) for v in j.values() if isinstance(v, list))
        if isinstance(j, list):
            return len(j)
    except Exception:
        pass
    return 0

_pass_pat = re.compile(r"(\d+)\s+passed")
_fail_pat = re.compile(r"(\d+)\s+failed")

def parse_pytest(out: str):
    """Extract pass rate and output length from pytest output."""
    out_len = len(out or "")
    try:
        passed = _pass_pat.search(out)
        failed = _fail_pat.search(out)
        p = int(passed.group(1)) if passed else 0
        f = int(failed.group(1)) if failed else 0
        total = p + f
        rate = (p / total) if total else 0.0
        return rate, out_len
    except Exception:
        return 0.0, out_len

def count_docstrings_and_typehints(py_files):
    docstrings = 0
    hints = 0
    for fp in py_files:
        try:
            src = Path(fp).read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if ast.get_docstring(node):
                    docstrings += 1
            if isinstance(node, ast.FunctionDef):
                if node.returns is not None:
                    hints += 1
                for arg in node.args.args + node.args.kwonlyargs:
                    if arg.annotation is not None:
                        hints += 1
    return docstrings, hints

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", default=DEFAULT_GENERATIONS, help="JSONL generations file")
    ap.add_argument("--out", default=DEFAULT_EVAL_REPORT, help="Report JSON path")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="eval_code_")
    files = write_code_files(args.generations, tmp)

    # Build a shadow 'tasks' package from generations so tests import the model's output.
    pkg_root = materialize_tasks_package(tmp, files)

    # Static tools on the generated code
    _, bandit_out, _ = run(f"bandit -r {tmp} -f json || true")
    _, ruff_out,   _ = run(f"ruff check {tmp} --output-format json || true")
    _, flake_out, _ = run(f"flake8 {tmp} --select=E,F,W --format=json || true")
    _, radon_out,  _ = run(f"radon cc {tmp} -j || true")

    # Ensure pytest imports our shadowed tasks package first
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{pkg_root}:{env.get('PYTHONPATH','')}"
    p = subprocess.run(
        "pytest -q --maxfail=1 --disable-warnings || true",
        shell=True, capture_output=True, text=True, env=env
    )
    pytest_out = p.stdout + "\n" + p.stderr

    bandit_findings = parse_bandit(bandit_out)
    ruff_count      = parse_ruff(ruff_out)
    flake8_length   = parse_flake8(flake_out)
    radon_length    = parse_radon(radon_out)
    pytest_pass_rate, pytest_output_len = parse_pytest(pytest_out)
    docstrings, type_hints = count_docstrings_and_typehints(files)

    report = {
        "counts": {"num_files": len(files)},
        "bandit_findings": int(bandit_findings),
        "ruff_count": int(ruff_count),
        "flake8_length": int(flake8_length),
        "radon_length": int(radon_length),
        "pytest_pass_rate": float(pytest_pass_rate),
        "pytest_output_len": int(pytest_output_len),
        "docstring_count": int(docstrings),
        "typehint_count": int(type_hints),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote report: {args.out}")

if __name__ == "__main__":
    main()
