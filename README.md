# MAVERICK: Validation Dataset and Analysis Code

> **MAVERICK: Breaking the FPR–FNR Seesaw in LLM Output Verification Through Strategy-Differentiated Multi-Agent Consensus**

## Contents

```
MAVERICK_deposit/
├── README.md
├── dataset/
│   ├── MAVERICK_validation_dataset.csv    # N=1,000 references with gold-standard labels
│   └── MAVERICK_validation_dataset.json   # Same data in JSON
└── code/
    └── reproduce_analysis.py              # Single entry point: reproduces all paper statistics
```

## Dataset

N=1,000 references in GB/T 7714-2025 format, each annotated with:

| Field | Description |
|-------|-------------|
| `id` | Reference number (1–1000) |
| `truth` | `genuine` or `non_genuine` |
| `category` | Subcategory (A1–A5 genuine, B1–B3 fabricated, C1–C3 tampered) |
| `doi` | DOI string (where applicable) |
| `reference` | Full GB/T 7714-2025 formatted citation |
| `source` | Verification source |

**Composition:** 550 genuine + 450 non-genuine (300 fabricated + 150 tampered).

## Reproducing Results

```bash
python code/reproduce_analysis.py
```

Outputs all statistics reported in the paper:
- **Fleiss' κ = 0.9704** — three-agent agreement (Landis & Koch: almost perfect)
- **System-level FPR = 0%** — 95% CI upper bound: 0.66% (Clopper-Pearson)
- **System-level FNR = 3.3%** — 18/550 escalated for human review
- **Error independence** — Φ coefficients, Fisher's exact tests, Monte Carlo power
- **Spearman-Brown prophecy** — diminishing reliability returns from k > 3
- **p³ theorem** — numerical verification (≤ 2.4×10⁻⁵)

Dependencies: `numpy`, `scipy`.

## License

MIT License.

## Citation

If you use this dataset or code, please cite the MAVERICK paper (DOI forthcoming).