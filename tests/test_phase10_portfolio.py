"""Phase 10 portfolio and release-documentation checks."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_contains_final_held_out_results_and_architecture() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "₹2,670.07" in readme
    assert "90.28%" in readme
    assert "45,009" in readme
    assert "```mermaid" in readme
    assert "FastAPI" in readme
    assert "Streamlit" in readme


def test_portfolio_assets_are_tracked_and_nonempty() -> None:
    assets = [
        ROOT / "docs/assets/dashboard_overview.png",
        ROOT / "docs/assets/model_accuracy_latency.png",
        ROOT / "docs/assets/final_test_actual_vs_predicted.png",
        ROOT / "docs/assets/final_test_interval_coverage.png",
    ]
    for asset in assets:
        assert asset.exists()
        assert asset.stat().st_size > 1_000


def test_phase10_documents_final_test_policy() -> None:
    phase10 = (ROOT / "PHASE_10.md").read_text()
    assert "first scored" in phase10.lower()
    assert "no post-test retuning" in phase10.lower()
    assert "72" in phase10


def test_release_version_and_phase10_make_target() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    makefile = (ROOT / "Makefile").read_text()
    assert project["version"] == "1.0.0"
    assert "phase10:" in makefile
