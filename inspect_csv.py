import pandas as pd

csv_path = "data/processed/cais_ttc_fixed_debug.csv"

df = pd.read_csv(csv_path)

print("Frames:", len(df))

print("\nCAIS mode counts:")
print(df["cais_mode"].value_counts(dropna=False))

print("\nCAIS reason counts:")
print(df["cais_reason_codes"].value_counts(dropna=False).head(20))

print("\nRaw warnings:")
print(df["raw_warning_level"].value_counts(dropna=False))

print("\nConfirmed warnings:")
print(df["confirmed_warning_level"].value_counts(dropna=False))

print("\nSelected target safety:")
print(df["selected_target_valid_for_safety"].value_counts(dropna=False))

print("\nSide / ego target distribution:")
print(df["target_in_ego_corridor"].value_counts(dropna=False))

bad_missing_ttc_cais = df[
    (df["cais_reason_codes"].astype(str).str.contains("valid_ttc_below_threshold", na=False)) &
    (
        (df["ttc_valid_for_safety"] != True) |
        (df["target_ttc_s"].isna())
    )
]

bad_high_ttc_cais = df[
    (df["cais_reason_codes"].astype(str).str.contains("valid_ttc_below_threshold", na=False)) &
    (
        pd.to_numeric(df["target_ttc_s"], errors="coerce")
        > pd.to_numeric(df["cais_ttc_threshold_s"], errors="coerce")
    )
]

side_warning = df[
    (df["target_in_ego_corridor"] == False) &
    (pd.to_numeric(df["target_relevance"], errors="coerce") < 0.5) &
    (df["raw_warning_level"].isin(["advisory", "warning", "strong_warning"]))
]

print("\nBAD: CAIS says valid_ttc_below_threshold while TTC missing/invalid:", len(bad_missing_ttc_cais))
print("BAD: CAIS says valid_ttc_below_threshold while TTC above threshold:", len(bad_high_ttc_cais))
print("BAD: Low-relevance side target generated FCW warning:", len(side_warning))

cols = [
    "frame_index",
    "selected_target_track_id",
    "selected_target_valid_for_safety",
    "target_in_ego_corridor",
    "target_relevance",
    "target_distance_m",
    "target_ttc_s",
    "ttc_valid_for_safety",
    "cais_mode",
    "cais_score",
    "cais_reason_codes",
    "cais_ttc_used_s",
    "cais_ttc_threshold_s",
    "cais_ttc_source_track_id",
    "raw_warning_level",
    "confirmed_warning_level",
    "warning_suppressed_reason",
]

print("\nRows where CAIS used TTC:")
print(
    df[df["cais_reason_codes"].astype(str).str.contains("valid_ttc_below_threshold", na=False)][cols]
    .head(30)
    .to_string(index=False)
)

print("\nTail rows:")
print(df[cols].tail(30).to_string(index=False))