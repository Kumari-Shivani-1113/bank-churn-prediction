# 🏦 Bank Customer Churn Prediction
### Data Science & Data Mining | The Great One Bank

> A multi-phase machine learning project to predict which bank customers are likely to churn (leave), enabling the bank to take proactive retention action across 100,000 customers.

---

## 📌 Project Overview

| | |
|---|---|
| **Dataset** | BankChurners for EDA 2024 |
| **Records** | 5,998 customers, 21 features |
| **Target** | Attrition Flag — Existing Customer / Attrited Customer |
| **Final Model** | Random Forest (Optimized via GridSearchCV) |
| **Final Accuracy** | 95.33% |

---

## 🔄 Project Phases

```
Phase 1 ──► Phase 2 ──────────────────────────────────────────────► Phase 3
  EDA        Preprocessing + Model Training                          Pre-Production Test
             ├─ Outlier treatment (IQR)                              Apply model to
             ├─ Label encoding                                       10 unknown records
             ├─ One-hot encoding                                     Verify 20/80 split
             ├─ Feature selection (Spearman corr)
             ├─ Decision Tree (baseline)
             ├─ Random Forest (recommended)
             └─ GridSearchCV (hyperparameter tuning)
```

---

## 📊 Dataset — BankChurners

| Feature | Description |
|---|---|
| `Attrition Flag` | **Target** — Existing Customer / Attrited Customer |
| `Customer Age` | Age of customer |
| `Gender` | M / F |
| `Card Category` | Blue (94.99%), Silver (4.40%), Gold (0.55%), Platinum |
| `Marital Status` | Single, Married, Divorced, Unknown |
| `Total Trans Ct` | Total transaction count (top predictor) |
| `Total Trans Amt` | Total transaction amount |
| `Total Relationship Count` | Number of bank products held |
| `Avg Utilization Ratio` | Credit utilization ratio |
| `Months on book` | Customer tenure |

**Class distribution:** ~87% Existing Customers, ~13% Attrited Customers

---

## 🔬 Phase 1 — Exploratory Data Analysis

- ✅ 0 missing values, 0 duplicates
- Distribution histograms for all 13 numeric features
- Boxplots for outlier detection
- Scatter plots for feature relationships
- Correlation heatmap (15×15)
- Categorical value distribution analysis

---

## ⚙️ Phase 2 — Preprocessing & Modeling

### Data Cleaning
- Dropped `CLIENTNUM` (ID column, no predictive value)
- Label encoded `Attrition Flag` and `Gender`
- **Outlier treatment** on 13 numeric columns using IQR Whisker method
- **One-hot encoded** Education Level, Marital Status, Income Category, Card Category

### Feature Selection
Top 7 features selected via **Spearman correlation** with target:
```python
selected_columns = ['Total Relationship Count', 'Total Trans Ct', 'Total Trans Amt',
                    'Total Ct Chng Q4 Q1', 'Avg Utilization Ratio', 'Months on book']
```

### Train / Test / Validation Split
```
70% Training → 15% Testing → 15% Validation
```

### Hypothesis Testing
```
Chi-Square Test: Card Category vs Marital Status
├─ Chi-square statistic: 33.86
├─ P-value: 9.46e-05 (significant)
└─ Cramér's V: 1.0 (strong association)
```

---

## 🤖 Model Results

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Decision Tree (baseline) | 92.89% | 67.24% | 75.00% |
| **Random Forest (recommended)** | **95.33%** | **96.85%** | **97.88%** |

### 5-Fold Cross-Validation (Decision Tree)
```
Scores: [0.914, 0.951, 0.560, 0.210, 0.325]
Mean:   59.21% — high variance → Decision Tree unstable
```

### GridSearchCV — Best Parameters (Random Forest)
```python
{
  'n_estimators':      200,
  'max_depth':         None,
  'min_samples_split': 10,
  'min_samples_leaf':  1
}
```

---

## 🎯 Phase 3 — Final Pre-Production Test

Applied the trained Random Forest model to **10 unknown customer records** (simulating real production scoring). The model correctly maintained the expected **~20% attrition ratio** (2 out of 10 customers predicted as attrited).

---

## 📸 Results Preview

![Notebook](screenshots/notebook_results.png)

---

## 📁 Files in This Repo

| File | Description |
|---|---|
| `bank_churn_prediction.ipynb` | Full cleaned notebook — EDA, preprocessing, modeling, evaluation |
| `data/BankChurners-for-EDA-2024.xlsx` | Dataset (5,998 rows, 21 features) |
| `screenshots/` | Results and visualizations |

---

## 🛠️ Tech Stack

**Languages & Libraries:** Python, pandas, numpy, scikit-learn, matplotlib, seaborn, scipy  
**ML Models:** Decision Tree, Random Forest, GridSearchCV  
**Techniques:** EDA, IQR Outlier Treatment, Label/One-Hot Encoding, Spearman Correlation, Chi-Square Test, Cramér's V, 5-Fold Cross-Validation, Hyperparameter Tuning  
**Environment:** Google Colab  

---

## 💼 Resume Description

> Built a multi-phase bank customer churn prediction system on 5,998 records (21 features). Performed full EDA, IQR outlier treatment, and feature selection via Spearman correlation. Trained Decision Tree and Random Forest classifiers; optimized Random Forest using GridSearchCV achieving 95.33% accuracy, 96.85% precision, and 97.88% recall. Conducted Chi-square and Cramér's V hypothesis testing on categorical features. Applied production model to classify 10 unknown records, correctly maintaining the expected 20% attrition ratio.

---

*IS-680/682 Project — Kumari Shivani *
