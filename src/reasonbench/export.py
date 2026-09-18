"""Machine-readable results: report.json, JUnit XML, and a scores CSV."""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from reasonbench.gate import GateResult
    from reasonbench.scoring import CriterionSummary, GroupSummary
    from reasonbench.storage import SampleRow, ScoreRow

SCHEMA_VERSION = 1


def _criterion_json(c: CriterionSummary) -> dict[str, Any]:
    out: dict[str, Any] = c.model_dump(mode="json")
    out["coverage"] = c.coverage
    return out


def _group_json(s: GroupSummary) -> dict[str, Any]:
    out: dict[str, Any] = s.model_dump(mode="json")
    out["criteria"] = [_criterion_json(c) for c in s.criteria]
    return out


def build_report(  # noqa: PLR0913
    *,
    run_dir: Path,
    prompt_id: str,
    prompt_title: str,
    group_by: tuple[str, ...],
    overall: GroupSummary,
    groups: list[GroupSummary],
    gate: GateResult,
    exit_code: int,
    complete: bool = True,
    stopped_reason: str | None = None,
) -> dict[str, Any]:
    """Assemble the JSON document a downstream job reads."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_dir": str(run_dir),
            "prompt_id": prompt_id,
            "prompt_title": prompt_title,
            "complete": complete,
            "stopped_reason": stopped_reason,
            "group_by": list(group_by),
        },
        "totals": {
            "n_samples": overall.n_samples,
            "n_failed": overall.n_failed,
            "n_scored": overall.n_scored,
            "failure_rate": (
                overall.n_failed / overall.n_samples if overall.n_samples else 0.0
            ),
            "weighted_mean": overall.weighted_mean,
            "weighted_stdev": overall.weighted_stdev,
            "total_cost_usd": overall.total_cost,
            "availability": overall.availability,
        },
        "criteria": [_criterion_json(c) for c in overall.criteria],
        "groups": [_group_json(s) for s in groups],
        "gate": {
            "configured": gate.configured,
            "passed": gate.passed,
            "n_passed": gate.n_passed,
            "n_failed": gate.n_failed,
            "n_skipped": gate.n_skipped,
            "assertions": [a.model_dump(mode="json") for a in gate.assertions],
        },
        "exit_code": exit_code,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write ``report.json``."""
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _case_suite(
    samples: list[SampleRow],
    collapsed: dict[tuple[str, str], float | None],
    criterion_id: str | None,
) -> ET.Element:
    suite = ET.Element("testsuite", name="reasonbench.cases")
    failures = errors = 0
    for row in samples:
        name = row.case_id or f"{row.variant_id}:{row.repeat}"
        case = ET.SubElement(
            suite,
            "testcase",
            classname=f"{row.prompt_id}.{row.variant_id}.{row.model.replace('/', '.')}",
            name=name,
            time=f"{row.latency_s:.3f}",
        )
        if not row.ok:
            errors += 1
            ET.SubElement(
                case, "error", type="SampleFailed", message=(row.error or "")[:400]
            )
            continue
        if criterion_id is None:
            continue
        value = collapsed.get((row.sample_id, criterion_id))
        if value is not None and value < 1.0:
            failures += 1
            failure = ET.SubElement(
                case,
                "failure",
                type="CaseFailed",
                message=f"{criterion_id} scored {value:.2f}",
            )
            failure.text = (row.output or "")[:2000]
    suite.set("tests", str(len(samples)))
    suite.set("failures", str(failures))
    suite.set("errors", str(errors))
    return suite


def _gate_suite(gate: GateResult) -> ET.Element:
    suite = ET.Element("testsuite", name="reasonbench.gate")
    for assertion in gate.assertions:
        case = ET.SubElement(
            suite, "testcase", classname="reasonbench.gate", name=assertion.id, time="0"
        )
        if assertion.status == "failed":
            ET.SubElement(case, "failure", type="GateBreach", message=assertion.reason)
        elif assertion.status == "skipped":
            ET.SubElement(case, "skipped", message=assertion.reason)
    suite.set("tests", str(len(gate.assertions)))
    suite.set("failures", str(gate.n_failed))
    suite.set("skipped", str(gate.n_skipped))
    return suite


def write_junit(
    path: Path,
    samples: list[SampleRow],
    collapsed: dict[tuple[str, str], float | None],
    gate: GateResult,
    *,
    case_criterion: str | None = None,
) -> None:
    """Write a JUnit file with a per-case suite and a per-assertion gate suite."""
    root = ET.Element("testsuites", name="reasonbench")
    root.append(_case_suite(samples, collapsed, case_criterion))
    if gate.configured:
        root.append(_gate_suite(gate))
    total = sum(int(s.get("tests", 0)) for s in root)
    root.set("tests", str(total))
    root.set("failures", str(sum(int(s.get("failures", 0)) for s in root)))
    root.set("errors", str(sum(int(s.get("errors", 0)) for s in root)))
    ET.indent(root)
    path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))


SCORE_FIELDS = [
    "sample_id",
    "case_id",
    "model",
    "variant_id",
    "temperature",
    "reasoning_effort",
    "repeat",
    "criterion_id",
    "kind",
    "target",
    "weight",
    "judge_repeat",
    "score",
    "normalized",
    "applicable",
    "reason",
]


def write_scores_csv(
    path: Path, samples: list[SampleRow], scores: list[ScoreRow]
) -> None:
    """Write one row per (sample, criterion, judge repeat)."""
    by_id = {s.sample_id: s for s in samples}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for row in scores:
            sample = by_id.get(row.sample_id)
            if sample is None:
                continue
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "case_id": sample.case_id,
                    "model": sample.model,
                    "variant_id": sample.variant_id,
                    "temperature": sample.temperature,
                    "reasoning_effort": sample.reasoning_effort,
                    "repeat": sample.repeat,
                    "criterion_id": row.criterion_id,
                    "kind": row.kind,
                    "target": row.target,
                    "weight": row.weight,
                    "judge_repeat": row.judge_repeat,
                    "score": row.score,
                    "normalized": row.normalized,
                    "applicable": row.applicable,
                    "reason": row.reason,
                }
            )
