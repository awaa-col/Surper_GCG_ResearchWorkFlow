from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditRule:
    key: str
    severity: str
    pattern: str
    message: str


@dataclass(frozen=True)
class AuditFinding:
    rule: str
    severity: str
    path: str
    line: int
    snippet: str
    message: str


RULES = [
    AuditRule(
        key="default_1b_model",
        severity="high",
        pattern=r"google/gemma-3-1b-it|gemma-3-1b",
        message="Default model still points at 1B.",
    ),
    AuditRule(
        key="hardcoded_l17_l23",
        severity="high",
        pattern=r"\bL17\b|\bL23\b|TARGET_LAYER\s*=\s*17|ABL_LAYERS\s*=\s*\[17,\s*23\]|GLOBAL_LAYERS\s*=\s*\[5,\s*11,\s*17,\s*23\]",
        message="Hardcoded historical layer prior detected.",
    ),
    AuditRule(
        key="fixed_extract_layer_17",
        severity="high",
        pattern=r"extract_layer.*default\s*=\s*17|--extract_layer.*17",
        message="Extraction defaults still assume layer 17.",
    ),
    AuditRule(
        key="late_band_prior",
        severity="medium",
        pattern=r"range\(\s*18\s*,\s*2[56]\s*\)|LATE_LAYERS\s*=|LATE_SAFE_LAYER\s*=|LATE_FAMILY_LAYER\s*=",
        message="Late-layer band is fixed rather than rediscovered.",
    ),
    AuditRule(
        key="old_pipeline_preset",
        severity="high",
        pattern=r"baseline_diagnosis|gate_discovery_bootstrap|theory_rebuild_bootstrap|mechanism_discovery_foundation|eval_calibration",
        message="Legacy preset or stage name still referenced.",
    ),
    AuditRule(
        key="exp19_dependency",
        severity="medium",
        pattern=r"from experiments\.exp_19_l17_l23_late_impact import|import experiments\.exp_19_l17_l23_late_impact",
        message="Experiment depends directly on historical Exp19 implementation.",
    ),
]


DEFAULT_GLOBS = [
    "experiments/**/*.py",
    "pipeline/**/*.py",
    "probes/**/*.py",
    "*.md",
    "*.ipynb",
]


def iter_files(root: Path, globs: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in globs:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            if path in seen:
                continue
            seen.add(path)
            files.append(path)
    return sorted(files)


def scan_file(path: Path, root: Path) -> list[AuditFinding]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")

    findings: list[AuditFinding] = []
    lines = text.splitlines()
    for rule in RULES:
        regex = re.compile(rule.pattern)
        for lineno, line in enumerate(lines, start=1):
            if regex.search(line):
                findings.append(
                    AuditFinding(
                        rule=rule.key,
                        severity=rule.severity,
                        path=str(path.relative_to(root)).replace("\\", "/"),
                        line=lineno,
                        snippet=line.strip()[:240],
                        message=rule.message,
                    )
                )
    return findings


def summarize(findings: list[AuditFinding]) -> dict[str, object]:
    by_severity: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    for item in findings:
        by_severity[item.severity] = by_severity.get(item.severity, 0) + 1
        by_rule[item.rule] = by_rule.get(item.rule, 0) + 1
    return {
        "total_findings": len(findings),
        "by_severity": by_severity,
        "by_rule": by_rule,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static audit for 12B migration risks and historical 1B priors."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings: list[AuditFinding] = []
    for path in iter_files(root, DEFAULT_GLOBS):
        findings.extend(scan_file(path, root))

    findings.sort(key=lambda item: (item.path, item.line, item.rule))
    report = {
        "root": str(root),
        "summary": summarize(findings),
        "findings": [asdict(item) for item in findings],
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output_path)
        return

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
