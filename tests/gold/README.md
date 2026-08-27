# Gold reference files

Frozen CSV outputs used to check weightpipe results against external tools
(R `survey`, R `weightflow`, R `sampler`, and [svy](https://svylab.com/svy)).

Python weighting gold (`*_svy.csv`) is generated with:

```bash
uv run --extra gold python tests/gold/generate_svy_gold.py
```

These files are for package tests. Users of weightpipe do not need them.
