# Gold / reference artifacts for cross-package validation.
#
# ## Always-on (committed CSVs)
# `*_samplics.csv` — frozen exports from samplics SampleWeight.
# Compared in `tests/test_gold_csv.py` (no optional install required).
# Regenerate:
#   uv run --extra gold python tests/gold/generate_samplics_gold.py
#
# ## Live samplics (optional extra `gold`)
# `tests/test_gold_samplics.py` re-runs samplics at test time.
# CI installs with `uv sync --all-groups --extra gold --locked`.
#
# ## R survey + weightflow
# Generate (requires R + packages):
#   Rscript tests/gold/generate_r_gold.R
# Produces `*_r_survey.csv` and optionally `*_weightflow.csv`.
# Compared in `tests/test_gold_r_packages.py` (CSV tests skip if files missing;
# live rpy2 tests skip without extra `r-gold` / R packages).
#
# Each gold/recovery test documents source and tolerances.
# Do not loosen tolerances to green CI — diagnose first.
