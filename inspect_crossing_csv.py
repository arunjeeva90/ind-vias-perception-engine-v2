import pandas as pd

csv_path = "data/processed/cutin_full_debug_v2.csv"
df = pd.read_csv(csv_path)


def truthy_series(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


crossing = df[~df.get("crossing_state", pd.Series(dtype=str)).astype(str).isin(["", "none", "nan"])]
valid = crossing[truthy_series(crossing.get("crossing_valid_for_safety", pd.Series(dtype=str)))]
invalid = crossing[~truthy_series(crossing.get("crossing_valid_for_safety", pd.Series(dtype=str)))]

print("Frames:", len(df))
print("Total crossing classifications:", len(crossing))
print("Valid crossing classifications:", len(valid))
print("Invalid/suppressed crossing classifications:", len(invalid))

print("\nCrossing reason-code counts:")
if "crossing_reason_codes" in df.columns:
    reasons = []
    for value in df["crossing_reason_codes"].dropna().astype(str):
        reasons.extend(reason.strip() for reason in value.split(",") if reason.strip())
    print(pd.Series(reasons).value_counts(dropna=False))

print("\nValid left_to_right / right_to_left counts:")
if "crossing_state" in valid.columns:
    print(valid["crossing_state"].value_counts(dropna=False))
