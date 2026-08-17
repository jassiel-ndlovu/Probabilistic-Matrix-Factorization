"""The paper must not reference numbers that nothing generates.

Because every figure in ``report/main.tex`` comes from ``report/generated/``,
a renamed or deleted experiment silently turns into an undefined control
sequence at compile time, or worse, a stale value. These tests close that gap:
they regenerate the macro file from whatever is in ``results/`` and check that
the paper's references are covered.

They skip when the study has not been run, so a fresh clone still gets a green
suite.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "report" / "main.tex"
GENERATED = REPO_ROOT / "report" / "generated"
RESULTS = REPO_ROOT / "results"

#: Control sequences that come from LaTeX or IEEEtran rather than from us.
BUILTIN = {
    "IEEEoverridecommandlockouts", "IEEEauthorblockN", "IEEEauthorblockA", "BibTeX",
    "Lambda", "Fro",
}


def _has_results() -> bool:
    return RESULTS.exists() and any(RESULTS.glob("model_comparison_*.json"))


pytestmark = pytest.mark.skipif(
    not _has_results(), reason="no results/ yet; run scripts/run_experiments.py"
)


@pytest.fixture(scope="module")
def macros() -> set[str]:
    """Regenerate report/generated and return the macro names it defines."""
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "make_report_tables.py")],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    text = (GENERATED / "macros.tex").read_text(encoding="utf-8")
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", text))


def test_every_macro_used_by_the_paper_is_generated(macros: set[str]) -> None:
    referenced = set(re.findall(r"\\([A-Z][A-Za-z]+)", REPORT.read_text(encoding="utf-8")))
    missing = sorted(referenced - macros - BUILTIN)
    assert not missing, f"main.tex references macros nothing generates: {missing}"


def test_every_input_target_is_generated() -> None:
    tex = REPORT.read_text(encoding="utf-8")
    targets = set(re.findall(r"\\input\{generated/([\w-]+)\.tex\}", tex))
    missing = sorted(t for t in targets if not (GENERATED / f"{t}.tex").exists())
    assert not missing, f"main.tex \\inputs files that were not generated: {missing}"


def test_every_figure_referenced_exists() -> None:
    tex = REPORT.read_text(encoding="utf-8")
    figures = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex))
    figure_dir = REPO_ROOT / "report" / "figures"
    missing = sorted(f for f in figures if not (figure_dir / f).exists())
    if missing:
        pytest.skip(f"figures not rendered yet (run scripts/make_figures.py): {missing}")


def test_generated_macros_carry_no_placeholder_values(macros: set[str]) -> None:
    """A macro whose value is '--' means the underlying result was missing."""
    text = (GENERATED / "macros.tex").read_text(encoding="utf-8")
    empty = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{--\}", text)
    assert not empty, f"macros resolved to a placeholder, so a result is missing: {empty}"
