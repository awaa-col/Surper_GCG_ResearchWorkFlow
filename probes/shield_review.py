from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .shield_audit import (
    AuditItem,
    AuditResult,
    ShieldGemmaAuditor,
    pick_top_audit_candidates,
    select_indices_by_risk,
    summarize_audit_results,
)


def build_audit_results(
    *,
    prompts: Sequence[str],
    responses: Sequence[str],
    source_file: str,
    source_path: str,
    truncate_response: int,
    auditor: ShieldGemmaAuditor | None = None,
    metas: Sequence[dict[str, Any] | None] | None = None,
    progress: str | bool | None = None,
) -> list[AuditResult]:
    auditor = auditor or ShieldGemmaAuditor()
    resolved_metas = list(metas) if metas is not None else [None] * len(responses)
    items = [
        AuditItem(
            prompt=prompt,
            response=response,
            source_file=source_file,
            source_path=source_path,
            meta=meta,
        )
        for prompt, response, meta in zip(prompts, responses, resolved_metas)
    ]
    return auditor.audit_items(
        items,
        truncate_response=truncate_response,
        progress=progress,
    )


def audit_responses(
    *,
    prompts: Sequence[str],
    responses: Sequence[str],
    source_file: str,
    source_path: str,
    truncate_response: int,
    auditor: ShieldGemmaAuditor | None = None,
    metas: Sequence[dict[str, Any] | None] | None = None,
    progress: str | bool | None = None,
    include_items: bool = True,
    include_top_candidates: bool = True,
    include_selection: bool = False,
    safe_max_prob: float = 0.35,
    unsafe_min_prob: float = 0.5,
    group_by_meta_key: str | None = None,
) -> dict[str, Any]:
    results = build_audit_results(
        prompts=prompts,
        responses=responses,
        source_file=source_file,
        source_path=source_path,
        truncate_response=truncate_response,
        auditor=auditor,
        metas=metas,
        progress=progress,
    )
    return summarize_audits(
        results,
        include_items=include_items,
        include_top_candidates=include_top_candidates,
        include_selection=include_selection,
        safe_max_prob=safe_max_prob,
        unsafe_min_prob=unsafe_min_prob,
        group_by_meta_key=group_by_meta_key,
    )


def summarize_audits(
    results: Sequence[AuditResult],
    *,
    include_items: bool = True,
    include_top_candidates: bool = True,
    include_selection: bool = False,
    safe_max_prob: float = 0.35,
    unsafe_min_prob: float = 0.5,
    group_by_meta_key: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "summary": summarize_audit_results(results),
    }
    if group_by_meta_key:
        groups: dict[str, list[AuditResult]] = defaultdict(list)
        for result in results:
            group_name = str((result.meta or {}).get(group_by_meta_key, "unknown"))
            groups[group_name].append(result)
        payload[f"summary_by_{group_by_meta_key}"] = {
            name: summarize_audit_results(group_results)
            for name, group_results in sorted(groups.items())
        }
    if include_top_candidates:
        payload["top_candidates"] = pick_top_audit_candidates(results)
    if include_selection:
        payload["selection"] = select_indices_by_risk(
            results,
            safe_max_prob=safe_max_prob,
            unsafe_min_prob=unsafe_min_prob,
        )
    if include_items:
        payload["items"] = [result.to_dict() for result in results]
    return payload
