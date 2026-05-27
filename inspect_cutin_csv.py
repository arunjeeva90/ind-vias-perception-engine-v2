import pandas as pd

csv_path = "data/processed/cutin_full_debug_v2.csv"
df = pd.read_csv(csv_path)


def truthy_series(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])

print("Frames:", len(df))

print("\nCut-in state counts:")
print(df["cutin_state"].value_counts(dropna=False))

print("\nSide state counts:")
print(df["side_state"].value_counts(dropna=False))

print("\nCut-in warning candidates:")
print(df["cutin_warning_candidate"].value_counts(dropna=False))

print("\nCut-in candidates by reason:")
if "cutin_reason_codes" in df.columns:
    print(df[df["cutin_warning_candidate"].astype(str).str.lower().eq("cut_in_risk")]["cutin_reason_codes"].value_counts(dropna=False))

print("\nCut-in confirmed:")
print(df["cutin_warning_confirmed"].value_counts(dropna=False))

print("\nRaw warning levels:")
print(df["raw_warning_level"].value_counts(dropna=False))

print("\nConfirmed warning levels:")
print(df["confirmed_warning_level"].value_counts(dropna=False))

cols = [
    "frame_index",
    "selected_target_track_id",
    "target_in_ego_corridor",
    "target_relevance",
    "target_distance_m",
    "side_state",
    "cutin_state",
    "ttc_lateral_s",
    "cutin_confidence",
    "cutin_warning_candidate",
    "cutin_warning_confirmed",
    "raw_warning_level",
    "confirmed_warning_level",
    "ego_motion_state",
    "yaw_confidence",
]

existing = [c for c in cols if c in df.columns]

print("\nRows with cut-in activity:")
active = df[
    df["cutin_state"].astype(str).isin(
        ["LEFT_CUT_IN", "RIGHT_CUT_IN", "IN_PATH", "left_cut_in", "right_cut_in", "in_path"]
    )
]
print(active[existing].head(50).to_string(index=False))

print("\nRows with cut-in warning candidate:")
warn = df[df["cutin_warning_candidate"].astype(str).str.lower().isin(["true", "cut_in_risk", "1"])]
print(warn[existing].head(50).to_string(index=False))

min_lateral_ttc_s = 0.4
confirmed = df[df["cutin_warning_confirmed"].astype(str).str.lower().eq("cut_in_risk")]
low_relevance_candidates = warn[pd.to_numeric(warn.get("target_relevance"), errors="coerce") < 0.5]
print("\nCut-in diagnostics:")
print("Candidates with target_relevance < 0.5:", len(low_relevance_candidates))
if "corridor_entry_confirmed" in df.columns:
    print(
        "BAD: confirmed cut-ins with corridor_entry_confirmed == false:",
        len(confirmed[~truthy_series(confirmed["corridor_entry_confirmed"])]),
    )
if "lateral_motion_stable" in df.columns:
    print(
        "BAD: confirmed cut-ins with lateral_motion_stable == false:",
        len(confirmed[~truthy_series(confirmed["lateral_motion_stable"])]),
    )
if "target_distance_valid_for_safety" in df.columns:
    print(
        "BAD: confirmed cut-ins with invalid distance:",
        len(confirmed[~truthy_series(confirmed["target_distance_valid_for_safety"])]),
    )
if "ttc_lateral_s" in df.columns:
    print(
        "BAD: confirmed cut-ins with ttc_lateral_s below min_lateral_ttc_s:",
        len(confirmed[pd.to_numeric(confirmed["ttc_lateral_s"], errors="coerce") < min_lateral_ttc_s]),
    )
if "crossing_state" in df.columns:
    crossing = df[~df["crossing_state"].astype(str).isin(["", "none", "nan"])]
    print("Pedestrian crossing candidates:", len(crossing))
    print("Crossing tracks by crossing_state:")
    print(crossing["crossing_state"].value_counts(dropna=False))
    if "crossing_valid_for_safety" in df.columns:
        valid_crossing = crossing[truthy_series(crossing["crossing_valid_for_safety"])]
        print("Valid crossing classifications:", len(valid_crossing))
        print("Invalid/suppressed crossing classifications:", len(crossing) - len(valid_crossing))
    if "crossing_reason_codes" in df.columns:
        crossing_reasons = []
        for value in df["crossing_reason_codes"].dropna().astype(str):
            crossing_reasons.extend(reason.strip() for reason in value.split(",") if reason.strip())
        print("Crossing reason-code counts:")
        print(pd.Series(crossing_reasons).value_counts(dropna=False))
