"""Deterministic trajectory deduplication, before synthetic sample generation."""

from __future__ import annotations

import pandas as pd


class FingerprintLeakageError(ValueError):
    """Cross-split copies cannot be resolved without changing the experiment."""

    def __init__(self, report: dict):
        super().__init__("Fingerprint leakage across splits; no automatic deduplication applied.")
        self.report = report


DEDUP_POLICY = "Within each split, retain the first (original_trajectory_id, trajectory_id) in lexical order; exclude other copies before synthetic generation. Cross-split copies fail."
DEDUP_COLUMNS = [
    "quality_eligible_for_dataset", "dataset_split", "is_exact_duplicate",
    "retained_trajectory_id", "duplicate_excluded_rows", "duplicate_excluded_normal_rows",
]


def deduplicate_trajectory_manifest(inputs: pd.DataFrame, assignments: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """Annotate every input; retain audit rows and never change user assignments.

    Equality uses the existing normalized content_fingerprint, not geometric
    similarity. All nonempty fingerprints are checked across splits, including
    quality-ineligible paths. Blank fingerprints denote entirely invalid paths.
    """
    required = ["user_id", "trajectory_id", "original_trajectory_id", "content_fingerprint",
                "eligible_for_dataset", "exclusion_reason", "cleaned_row_count", "eligible_normal_row_count"]
    if any(c not in inputs for c in required):
        raise ValueError("Deduplication manifest is missing required columns.")
    if inputs["trajectory_id"].isna().any() or inputs["trajectory_id"].duplicated().any():
        raise ValueError("Duplicate or missing trajectory ID in deduplication manifest.")
    if inputs["content_fingerprint"].isna().any():
        raise ValueError("Missing trajectory fingerprint.")
    result = inputs.copy(deep=True).reset_index(drop=True)
    result["dataset_split"] = result["user_id"].map(assignments)
    if result["dataset_split"].isna().any() or not result["dataset_split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Missing or invalid user split assignment.")
    groups = []
    for fingerprint, group in result.loc[result["content_fingerprint"].ne("")].groupby("content_fingerprint", sort=True):
        if len(group) < 2:
            continue
        ordered = group.sort_values(["original_trajectory_id", "trajectory_id"], kind="stable")
        records = [{"user_id": str(r.user_id), "trajectory_id": str(r.trajectory_id),
                    "original_trajectory_id": str(r.original_trajectory_id),
                    "split": str(r.dataset_split), "rows": int(r.cleaned_row_count),
                    "quality_eligible_for_dataset": bool(r.eligible_for_dataset)}
                   for r in ordered.itertuples()]
        groups.append({"duplicate_fingerprint": str(fingerprint), "trajectories": records,
                       "cross_split": group["dataset_split"].nunique() > 1})
    report = {
        "policy": DEDUP_POLICY, "duplicate_fingerprint_count": len(groups),
        "duplicate_trajectory_count": sum(len(g["trajectories"]) for g in groups),
        "cross_split_duplicate_fingerprint_count": sum(g["cross_split"] for g in groups),
        "groups": groups, "duplicate_exclusions": [], "excluded_duplicate_trajectory_count": 0,
        "excluded_duplicate_rows": 0, "excluded_duplicate_normal_rows": 0,
        "normalization": "Existing SHA256 of lat/lon 7 decimal places and elapsed nanoseconds.",
    }
    # Check every group before changing any eligibility, even within-split groups.
    if report["cross_split_duplicate_fingerprint_count"]:
        report["ready_for_stage5_b"] = False
        raise FingerprintLeakageError(report)
    result["quality_eligible_for_dataset"] = result["eligible_for_dataset"].astype(bool)
    result["is_exact_duplicate"] = False
    result["retained_trajectory_id"] = ""
    result["duplicate_excluded_rows"] = 0
    result["duplicate_excluded_normal_rows"] = 0
    for group in groups:
        retained = group["trajectories"][0]
        group.update(split=retained["split"], retained_trajectory_id=retained["trajectory_id"])
        fingerprint = group["duplicate_fingerprint"]
        members = result["content_fingerprint"].eq(fingerprint)
        result.loc[members, "retained_trajectory_id"] = retained["trajectory_id"]
        excluded = members & result["trajectory_id"].ne(retained["trajectory_id"])
        for row in result.loc[excluded].itertuples():
            normal_rows = int(row.eligible_normal_row_count) if row.quality_eligible_for_dataset else 0
            record = {
                "duplicate_fingerprint": fingerprint, "user_id": str(row.user_id),
                "trajectory_id": str(row.trajectory_id), "split": str(row.dataset_split),
                "retained_user_id": retained["user_id"],
                "retained_trajectory_id": retained["trajectory_id"],
                "excluded_trajectory_id": str(row.trajectory_id),
                "exclusion_reason": "exact_duplicate_trajectory",
                "excluded_rows": int(row.cleaned_row_count), "excluded_normal_rows": normal_rows,
            }
            report["duplicate_exclusions"].append(record)
            selected = result["trajectory_id"].eq(row.trajectory_id)
            result.loc[selected, "duplicate_excluded_rows"] = record["excluded_rows"]
            result.loc[selected, "duplicate_excluded_normal_rows"] = normal_rows
        result.loc[excluded, "is_exact_duplicate"] = True
        result.loc[excluded, "eligible_for_dataset"] = False
        result.loc[excluded, "exclusion_reason"] = result.loc[excluded, "exclusion_reason"].fillna("").map(
            lambda reason: ";".join(filter(None, [reason, "exact_duplicate_trajectory"])))
    report["excluded_duplicate_trajectory_count"] = len(report["duplicate_exclusions"])
    report["excluded_duplicate_rows"] = sum(r["excluded_rows"] for r in report["duplicate_exclusions"])
    report["excluded_duplicate_normal_rows"] = sum(r["excluded_normal_rows"] for r in report["duplicate_exclusions"])
    report["remaining_model_duplicate_fingerprint_count"] = 0
    report["ready_for_stage5_b"] = True
    return result, report
