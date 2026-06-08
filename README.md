# Liesen-Gai Causal Inference Final Project

Final project for a causal inference course: estimating the causal effect of physical activity on cardiovascular disease (CVD) using the Ulianova Kaggle dataset, with positivity diagnostics, covariate balance, BMI mediation, and sensitivity analyses.

## Repository structure

```
Liesen-Gai-Causal-Inference-Final-Project/
├── README.md
├── code/
│   └── analysis.py                 # Main analysis pipeline (reproduces all results)
├── data/
│   └── cardio_train.csv            # Ulianova dataset (70,000 raw records; analysis input)
├── output/
│   ├── summary.json                # Summary metrics
│   ├── report_metrics.json         # Formatted values for the report
│   ├── estimates.csv               # ACE estimates (all methods)
│   ├── balance.csv                 # Covariate balance (SMD)
│   ├── heterogeneity.csv           # Age-stratified IPW results
│   ├── adjustment_sensitivity.csv  # Adjustment-set sensitivity
│   └── mediation_sensitivity.csv   # Mediation sensitivity scan
├── figures/
│   ├── forest_plot.pdf             # Forest plot of ACE estimates
│   ├── ps_overlap.pdf              # Propensity score overlap (positivity)
│   ├── love_plot.pdf               # Covariate balance (love plot)
│   ├── mediation_decomp.pdf        # BMI mediation decomposition
│   ├── mediation_sensitivity.pdf   # Mediation rho sensitivity
│   ├── age_heterogeneity.pdf       # Age-stratified effects
│   ├── effect_modification.pdf     # Smoking effect modification
│   └── adjustment_sensitivity.pdf  # Adjustment-set sensitivity
└── report/
    ├── final_pj_en.tex             # English report source (pdfLaTeX)
    ├── final_pj_en.pdf             # Compiled English report
    └── references.bib              # BibTeX bibliography (all cited references)
```

**Note:** This repository contains the English report and all materials needed to reproduce the analysis. The analysis uses `data/cardio_train.csv` only.

## Requirements

- Python 3.10+
- Python packages: `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `matplotlib`
- TeX Live with `pdflatex`

## Clone and reproduce

```bash
git clone https://github.com/daomuyang/Liesen-Gai-Causal-Inference-Final-Project.git
cd Liesen-Gai-Causal-Inference-Final-Project

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas numpy statsmodels scikit-learn matplotlib

MPLCONFIGDIR=.mplconfig python code/analysis.py
```

## Compile the report

```bash
cd report
pdflatex final_pj_en.tex
pdflatex final_pj_en.tex
```

## Data cleaning

Keep rows with `ap_hi` in [80, 200], `ap_lo` in [40, 120], and `ap_hi > ap_lo`; BMI in [15, 60]; height in [140, 220] cm. This removes 1,608 implausible records, yielding **N = 68,392**.

## Data source

Ulianova cardiovascular dataset: [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
