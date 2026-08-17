# MAVERICK Cross-Domain Validation Dataset (N=400)

**Companion dataset for the Extended Data cross-domain validation experiments of the paper**
*MAVERICK: Breaking the FPR-FNR Seesaw in LLM Output Verification Through Strategy-Differentiated Multi-Agent Consensus* (Nature Machine Intelligence submission).

Main validation dataset (N=5094): DOI 10.5281/zenodo.21975064.

## 1. Files

| File | Description |
|------|-------------|
| `MAVERICK_crossdomain_dataset_400.csv` | Tabular export (UTF-8), one row per record |
| `MAVERICK_crossdomain_dataset_400.json` | Same data as JSON (indented, UTF-8) with label legend |
| `README.md` | This documentation |

## 2. Composition

| Source | ID prefix | N | genuine | manipulated | in_the_wild |
|--------|-----------|---|---------|-------------|-------------|
| US legal | US-L-001~100 | 100 | 50 | 30 | 20 |
| CN legal | CN-L-001~100 | 100 | 50 | 30 | 20 |
| US clinical | US-C-001~100 | 100 | 50 | 30 | 20 |
| CN clinical | CN-C-001~100 | 100 | 50 | 30 | 20 |
| **Total** | — | **400** | **200** | **120** | **80** |

Genuine / Manipulated / In-the-Wild = 50% / 30% / 20%, matching the main 5094 dataset.

## 3. Gold standard

Programmatically determined via authoritative registry primary-key lookup
(deterministic, versioned): genuine — primary key hits the registry; manipulated —
single-field tampering → key not found → false; in-the-wild — synthetic-wild
hallucinated samples → key not found → false. All four sources passed 100/100 checks.

## 4. Reproducibility

Experiment code (`run_crossdomain_experiment.py`, thresholds fixed at the main-paper
values, not tuned) and full per-sample agent verdicts / metrics are available in the
paper's GitHub repository (`code/` and `results/`).
