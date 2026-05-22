# ML-vs-Actuarial-Methods-Mortality
A comparison of traditional Actuarial methods against Machine Learning methods in Mortality/Lapse rate modelling. This project uses mortality data from the Human Mortality Database, namely the Australian subset from 1921 to 2021.

---

## Motivation

A core component of life insurance pricing, annuity valuation and pensionfund management comes from mortality modelling, where the industry standard for decades have been actuarial models such as Lee-carter (1992) and the Cairns-Blade-Dowd model (CBD model, 2006). This study aims to compare these traditional models to the machine learning methods that have been on the rise during recent years, answering the question: **Can modern ML methods outperform the models built on decades of actuarial intuition?**

This project compares the Lee-Carter, CBD models to ML methods XGBoost and Neural Networks on an out of sample forecasting task. Then, a hybrid model was constructed, combining the strengths of both actuarial and machine learning approaches.

---

## Data

- **Source:** [Human Mortality Database](https://www.mortality.org) — Australia, 1x1 (single year of age, single calendar year)
- **Period:** 1921–2021
- **Train/test split:** Train on 1921–2000, test on 2001–2021 (temporal split, no data leakage)
- **Target:** Log mortality rate log(m_x,t) for ages 0–100

---

## Models

### Classical Actuarial

| Model | Description |
|-------|-------------|
| **Lee-Carter (1992)** | Period model using SVD decomposition. Fits age-specific mortality shape (α_x, β_x) and a single time trend (κ_t). The industry standard for over 30 years. |
| **Cairns-Blake-Dowd (2006)** | Two-factor model designed for older ages (60–89). Models logit(q_x,t) as a linear function of centred age, with two time-varying parameters capturing mortality level and age slope. |

### Machine Learning

| Model | Description |
|-------|-------------|
| **XGBoost** | Gradient boosted trees with feature engineering (log age, age², age-year interaction). Represents the standard ML baseline. |
| **Neural Network** | Feedforward network (4→64→32→16→1) trained with Adam optimiser. Learns smooth non-linear relationships between age, year, and log mortality. |

### Hybrid

| Model | Description |
|-------|-------------|
| **LC + NN Hybrid** | Soft-blended model using a sigmoid transition between Lee-Carter (young ages) and Neural Network (older ages). Tested at cutoff ages 35, 45, and 55. |


---

## Key Initial Results

Out-of-sample performance (2001–2021):

| Model | Test MAE | Test RMSE | Age Range |
|-------|----------|-----------|-----------|
| Lee-Carter | 0.2367 | 0.2896 | 0–100 |
| XGBoost | 0.2671 | 0.3370 | 0–100 |
| Neural Network | 0.1769 | 0.2679 | 0–100 |
| CBD | 0.1972 | 0.2193 | 60–89 |
| Hybrid (cutoff=35) | 0.1025 | 0.1568 | 0–100 |
| **Hybrid (cutoff=40)** | **0.1001** | **0.1539** | 0–100 |
| Hybrid (cutoff=45) | 0.1037 | 0.1567 | 0–100 |
| Hybrid (cutoff=55) | 0.1205 | 0.1687 | 0–100 |

The best hybrid model achieves a **57% reduction in MAE** versus Lee-Carter alone.

### Error by Age

A core finding of this study was how performance tended to fluctuate based on the age. 

![Error by Age](results/Prediction%20Error%20by%20Age%20Comparative%20Graph.png)

- **Ages 0–25:** Lee Carter handles the accident hump the best, where ML methods were unable to capture the spike in young male mortality driven by behavioural factors.
- **Ages 30–45:** Here, models converged in performance, but Lee-Carter still showed the lowest Error overall
- **Ages 45–85:** Lee-Carter deteriorates significantly; Neural Networks now show a very low prediction error, near zero.
- **Ages 60–89:** CBD sits between Lee-Carter and Neural Net in its designed age range, error decreased significantly for CBD at ages ~85.

This suggests a natural **hybrid approach** — Lee-Carter for young ages, Neural Network for ages 45+ — as a direction for future work.

---

## Hybrid Model


Motivated by the error-by-age analysis showing Lee-Carter outperforming at young ages and the Neural Network dominating at older ages, a soft-blended hybrid model was constructed using a sigmoid transition:
```
weight_nn(x) = 1 / (1 + exp(-0.3 * (x - cutoff)))
prediction(x) = (1 - weight_nn) * LC_pred + weight_nn * NN_pred
```

This produces a smooth continuous mortality curve that naturally transitions from Lee-Carter at young ages to Neural Network at older ages — combining actuarial interpretability with ML predictive power.

The optimal cutoff at age 40 performs best, consistent with the error-by-age analysis showing the Neural Network begins outperforming Lee-Carter at around age 30–40. This represents a **58% reduction in MAE** versus Lee-Carter alone.

---

### Repo Structure

```
data/                   # HMD mortality data (not tracked in git)
notebooks/
  01_eda.ipynb          # Exploratory data analysis
  02_lee_carter.ipynb   # Lee-Carter model
  03_xgboost.ipynb      # XGBoost model
  04_neuralnet.ipynb    # Neural Network model
  05_CBD.ipynb          # CBD Model
  06_comparisons.ipynb  # Full model comparison and visualisations
  07_lee_carter_NN_hybrid.ipynb # hybrid model
requirements.txt
README.md
```

---

## Setup

```bash
# clone the repo
git clone https://github.com/yourusername/ML-vs-Actuarial-Methods-Mortality.git
cd ML-vs-Actuarial-Methods-Mortality

# create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# install dependencies
pip install -r requirements.txt
```

**Data:** Download `Deaths_1x1.txt` and `Exposures_1x1.txt` for Australia from [mortality.org](https://www.mortality.org) and place them in the `data/` folder.

---

## Future Work

- **Sex-disaggregated modelling:** Separate models for male and female mortality capture the gender gap more precisely
- **Multi-country comparison:** Extend to UK, US, and Japan to test generalisability
- **Cause-of-death features:** Adding cause-specific mortality data could help models learn the accident hump
- **Deeper architectures:** LSTM or transformer models to explicitly capture cohort effects visible in Lee-Carter residuals

---

## Background

This project was built as part of a double degree in Actuarial Studies and Computer Science at UNSW Sydney, combining coursework in survival modelling, financial mathematics, and machine learning.
