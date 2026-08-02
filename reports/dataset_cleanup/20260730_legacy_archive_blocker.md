# Legacy archive blocker — 2026-07-30

## Required archive

Requested destination:

```text
/home/vicharak/Mobility_ADAS/ADVIS/DMS/legacy_dataset_archive/
legacy_eye_datasets_20260730.tar.zst
```

Result: **not created**.

The legacy payload was untracked/ignored local data. It was never part of the
Git repository or the authoritative reviewed handoff.

## Blocking condition

The only quarantined payload was moved to the following temporary path during
the previous cleanup:

```text
/tmp/dms_legacy_eye_datasets_20260730.FM4eIG/datasets/
```

That path is now absent. `/tmp` is backed by `tmpfs`, and it was purged between
turns before a durable archive was requested and created.

The active repository still correctly contains no legacy eye datasets. The
retained inventory contains 10,576 path/hash records with 6,177 unique hashes:

```text
reports/dataset_cleanup/legacy_eye_datasets_before_removal.sha256
```

## Recovery audit

| Check | Result |
|---|---|
| Original `/tmp` quarantine | absent |
| Alternate `/tmp/dms_legacy_eye_datasets*` | none |
| Legacy dataset roots under `/home/vicharak` | none |
| Representative filenames under `/home/vicharak` | none |
| Matching unique hashes in authoritative handoff eye data | 0 / 6,177 |
| Matching unique hashes in eye-related user Trash data | 0 / 6,177 |

The authoritative handoff is intact and remains valid for new training, but it
is not a byte-identical backup of these obsolete legacy files.

The temporary quarantine was purged before durable archival, and the files are
no longer recoverable from the AXON filesystem. No filesystem-level recovery,
undelete, disk carving, reconstruction, or substitution is authorized.

The retained 10,576-entry SHA-256 inventory is historical evidence only. It
must not be used to claim that any legacy image has been recovered. Because
there are zero hash matches, the authoritative handoff must not be used to
recreate the old datasets.

Any copy later found in the original Windows workspace must be archived outside
the repository, verified against the historical inventory, and never mixed
into authoritative handoff training.

## Stop decision

Execution stopped before phone-frame investigation or any model training,
environment installation, export, RKNN conversion, runtime replacement,
commit, or push.

The user has acknowledged this incident as non-blocking for new model
development. Training may continue only from the authoritative reviewed
handoff sources.
