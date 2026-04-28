# IKKA: Inversion Classification via Critical Anomalies

Topologically motivated anomaly-weighting framework for robust visual servoing under distribution shift.

> **Paper:** *IKKA: Inversion Classification via Critical Anomalies for Robust Visual Servoing*
> Under review at NeurIPS 2026. Preprint: [arXiv:2604.08754](https://arxiv.org/abs/2604.08754)

## Overview

IKKA introduces a topological anomaly weight

$$W(x) = E(x) \cdot T(x) \cdot M(x)$$

where:
- **E(x)** — local extremality (error magnitude + tracker confidence drop)
- **T(x)** — boundary transversality (gradient non-alignment across class probabilities)
- **M(x)** — multi-scale persistence (sublevel-set H₁ persistent homology of the error time-series, with Cohen–Steiner stability)

The weight modulates control updates near ambiguous decision regions. On a 230-run Raspberry Pi 4 benchmark, IKKA reduces the 95th-percentile lateral error by 24 % under stress conditions while increasing throughput from 20.0 to 24.8 Hz.

## Repository structure

```
ikka/
├── weight.py          # W(x) = E(x) · T(x) · M(x)
├── extremality.py     # E component
├── transversality.py  # T component
├── persistence.py     # M component (sublevel-set H₁)
└── control.py         # bounded IBVS yaw-rate command

tests/
└── test_weight.py     # smoke tests

analysis.py            # benchmark replay over manifest.csv
manifest.csv           # 230-run experiment manifest
requirements.txt
LICENSE                # MIT
```

## Installation

```bash
git clone https://github.com/<owner>/ikka.git
cd ikka
pip install -r requirements.txt
```

## Reproducing the benchmark

```bash
python analysis.py --manifest manifest.csv \
    --input out_logs/ \
    --output artefacts/
```

This regenerates the figures and statistics in the paper from the per-run logs.

## Counterexample (IKKA vs. SVM)

The synthetic three-class counterexample (Section 6 of the paper) is reproducible via:

```bash
python -m ikka.examples.svm_vs_ikka
```

It constructs a Wada-like triple-boundary junction at the origin and shows that IKKA identifies points an order of magnitude closer to the topologically indispensable junction than SVM support vectors.

## Hardware

- Raspberry Pi 4B (4 GB RAM)
- Pi Camera Module V2
- QVGA (320 × 240 px), CPU only
- IKKA per-frame overhead: ≈ 1.4 ms

## Citation

```bibtex
@article{ikka2026,
  title   = {IKKA: Inversion Classification via Critical Anomalies for Robust Visual Servoing},
  author  = {Anonymous},
  journal = {arXiv preprint arXiv:2604.08754},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.
