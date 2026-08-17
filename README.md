# MAVERICK: Validation Dataset and Analysis Code (v8, N=5094)

> **MAVERICK: Breaking the FPR–FNR Seesaw in LLM Output Verification Through Strategy-Differentiated Multi-Agent Consensus**

## Contents

```
MAVERICK_deposit/
├── README.md
├── dataset/
│   ├── MAVERICK_validation_dataset.csv    # N=5,094 references with gold-standard labels
│   ├── MAVERICK_validation_dataset.json   # Same data in JSON
│   └── legacy_v1_N1000/                   # Archived v1 dataset (N=1,000), kept for reference
└── code/
    ├── reproduce_analysis.py              # Single entry point: reproduces all paper statistics
    └── convert_master_to_deposit.py       # Master dataset (jsonl) → deposit CSV/JSON converter
```

## Dataset

N=5,094 references, each annotated with:

| Field | Description |
|-------|-------------|
| `id` | Unique reference ID (M0000–M5093) |
| `truth` | `genuine` or `non_genuine` |
| `category` | `authentic` / `fabricated` / `tampered` / `wild` |
| `subtype` | Finer subcategory (where applicable) |
| `doi` | DOI string (where applicable) |
| `reference` | Full citation, verbatim |
| `source` | Verification source / notes |

**Composition (gold labels):**

| Category | Count |
|----------|-------|
| authentic | 2,788 |
| fabricated | 1,280 |
| tampered | 602 |
| wild | 424 |
| **Total** | **5,094** |

## Results (from the v8 summary report, 26 verification batches)

- **Unambiguous verdict rate: 93.4%** (4,757/5,094; genuine 2,670 + fabricated 1,585 + tampered 502)
- **Fabricated-reference detection: 100%** (1,280/1,280, zero misses)
- **System-level FPR: 0%** — 95% CI upper bound **0.059%** (Clopper-Pearson, N=5,094)
- **System-level FNR ≈ 0%** (verifier-error definition)
- **p³ theorem**: FPR_sys = p_A×p_B×p_C = 2.4×10⁻⁵ under conditional independence; worst-case correlated bound 3.4×10⁻⁵
- **Gaussian copula sensitivity**: ρ=0.1 → 3.7×, ρ=0.3 → 21.7× inflation vs independence

## Reproducing Results

```bash
python code/reproduce_analysis.py
```

Outputs all statistics reported in the paper (v8):
- Dataset composition (read directly from the released CSV)
- Verifier verdict distribution
- System-level FPR / FNR with Clopper-Pearson 95% CI
- p³ theorem numerical verification
- Error-correlation (Gaussian copula) sensitivity analysis

Dependencies: `numpy`, `scipy`.

## License

MIT License.

## Citation

If you use this dataset or code, please cite the MAVERICK paper. Dataset deposited at Zenodo: [10.5281/zenodo.21975064](https://doi.org/10.5281/zenodo.21975064).
