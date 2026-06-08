# Liesen-Gai Causal Inference Final Project

Final project for a causal inference course: estimating the causal effect of physical activity on cardiovascular disease (CVD) using the Ulianova Kaggle dataset, with positivity diagnostics, covariate balance, BMI mediation, and sensitivity analyses.

## Repository structure

```
├── data/
│   └── cardio_train.csv          # Ulianova dataset (70,000 raw records)
├── code/
│   └── analysis.py               # Main reproducible pipeline
├── output/
│   ├── summary.json              # Summary metrics
│   ├── report_metrics.json       # Formatted values for the report
│   └── ...                       # Other CSV/JSON outputs
├── figures/                      # Report figures (PDF)
├── report/
│   └── final_pj_en.tex           # English report source (pdfLaTeX)
└── FinalPJ_Liesen_Gai.pdf        # Compiled English report
```

**Note:** Legacy UCI Heart Disease files under `data/` (if present locally) are not used in this analysis.

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
