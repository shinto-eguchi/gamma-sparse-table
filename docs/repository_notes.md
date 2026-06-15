# Repository notes

The uploaded working notebook was reorganized into a small Python package plus reproducible scripts.  The public notebook is intentionally lightweight: it imports the package and demonstrates a small high-leverage stress run.  The exploratory blocks for validation selection, stability diagnostics, and WJ selection were folded into reusable functions where appropriate.

For the paper, the most important public entry point is:

```bash
python scripts/run_high_leverage_stress.py --n-reps 60 --b-resample 10 --n-jobs 7
```

The old spelling `gamma_sparce3.ipynb` was not used in file names; the public-facing spelling is `gamma_sparse`.
