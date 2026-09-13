"""Stage 12 final repository audit.

Checks that required sources, measured artifacts, and WRITEUP sections exist,
then runs the correctness tests. Writes results/stage12_audit.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

REQUIRED_SOURCES = [
    "src/__init__.py",
    "src/model_wrapper.py",
    "src/cache_manager.py",
    "src/cache_utils.py",
    "src/evictions.py",
    "src/position_utils.py",
    "src/evaluation.py",
    "benchmark.py",
    "evaluate.py",
    "visualise.py",
    "stage4_experiment.py",
    "test_sliding_window.py",
    "test_attention_sink.py",
    "test_heavy_hitter.py",
    "test_position_handling.py",
    "test_correctness_harness.py",
    "test_correctness.py",
    "test_evaluation.py",
    "README.md",
    "PROJECT_STATUS.md",
    "WRITEUP.md",
    "requirements.txt",
    ".gitignore",
    "audit_repository.py",
]

REQUIRED_ARTIFACTS = [
    "results/plots/attention/attention_metrics.json",
    "results/plots/attention/attention_arrays.npz",
    "results/plots/attention/attention_by_token_position.png",
    "results/plots/attention/early_token_attention.png",
    "results/plots/attention/layer_attention_heatmap.png",
    "results/plots/attention/head_attention_heatmap.png",
    "results/plots/quality/perplexity_vs_budget.png",
    "results/plots/quality/nih_accuracy_vs_budget.png",
    "results/plots/memory/peak_memory_vs_budget.png",
    "results/plots/memory/theoretical_kv_vs_budget.png",
    "results/plots/stage10_plot_index.json",
    "results/quality/budget_16/stage9_perplexity.json",
    "results/quality/budget_16/stage9_nih.json",
    "results/quality/budget_32/stage9_perplexity.json",
    "results/quality/budget_32/stage9_nih.json",
    "results/benchmark/budget_16/benchmark_summary.json",
    "results/benchmark/budget_32/benchmark_summary.json",
    "results/benchmark/budget_64/benchmark_summary.json",
    "results/stage4_sliding_vs_streaming.json",
    "results/stage7_quality_comparison.json",
]

WRITEUP_SECTIONS = [
    "1. Motivation",
    "2. Attention sinks",
    "3. Sliding Window",
    "4. StreamingLLM",
    "5. H2O",
    "6. RoPE",
    "7. Experimental setup",
    "8. Perplexity results",
    "9. NIH results",
    "10. Memory results",
    "11. Quality-vs-memory",
    "12. H2O early-token bias",
    "13. Recommendation for multi-turn agents",
    "14. Recommendation for long-document summarizers",
    "15. Limitations",
]

FAST_TESTS = [
    "test_sliding_window.py",
    "test_attention_sink.py",
    "test_heavy_hitter.py",
    "test_correctness_harness.py",
    "test_evaluation.py",
]

MODEL_TESTS = [
    "test_position_handling.py",
    "test_correctness.py",
]


def missing(paths: list[str]) -> list[str]:
    return [path for path in paths if not (ROOT / path).is_file()]


def writeup_missing_sections() -> list[str]:
    text = (ROOT / "WRITEUP.md").read_text(encoding="utf-8")
    return [section for section in WRITEUP_SECTIONS if section not in text]


def run_script(name: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(ROOT / name)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "script": name,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout_tail": completed.stdout[-500:],
        "stderr_tail": completed.stderr[-500:],
    }


def main() -> int:
    report = {
        "stage": 12,
        "missing_sources": missing(REQUIRED_SOURCES),
        "missing_artifacts": missing(REQUIRED_ARTIFACTS),
        "missing_writeup_sections": writeup_missing_sections(),
        "fast_tests": [],
        "model_tests": [],
    }

    ok = (
        not report["missing_sources"]
        and not report["missing_artifacts"]
        and not report["missing_writeup_sections"]
    )

    for name in FAST_TESTS:
        result = run_script(name)
        report["fast_tests"].append(result)
        ok = ok and bool(result["passed"])

    for name in MODEL_TESTS:
        result = run_script(name)
        report["model_tests"].append(result)
        ok = ok and bool(result["passed"])

    report["passed"] = ok
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "stage12_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({key: report[key] for key in (
        "missing_sources",
        "missing_artifacts",
        "missing_writeup_sections",
        "passed",
    )}, indent=2))
    print(f"Wrote {output}")
    if not ok:
        for group in ("fast_tests", "model_tests"):
            for result in report[group]:
                if not result["passed"]:
                    print(f"FAIL {result['script']} rc={result['returncode']}")
                    if result["stderr_tail"]:
                        print(result["stderr_tail"])
        return 1
    print("Stage 12 repository audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
