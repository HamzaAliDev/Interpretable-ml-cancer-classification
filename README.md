# Breast Cancer Classification and Explainability

This project builds and evaluates classical machine learning models for breast cancer diagnosis support using the Wisconsin Breast Cancer dataset. It includes rigorous model evaluation via 5-fold cross-validation and SHAP-based interpretability to explain feature-level impact on predictions.

## Project Highlights

- End-to-end notebook workflow from data loading to explainability
- Comparison of three classifiers: Random Forest, SVM, and XGBoost
- Robust evaluation using 5-fold cross-validation - SVM and XGBoost achieve 97.36% +/- 1.47% mean accuracy
- SHAP visual reports saved in the reports folder

## Project Structure

```text
ml-health/
|-- data/
|   `-- breast_cancer.csv
|-- notebooks/
|   |-- data_loading.ipynb
|   |-- step2_eda.ipynb
|   |-- step3_preprocessing.ipynb
|   |-- step4_model_training.ipynb
|   `-- step5_shap_interpretability.ipynb
|-- reports/
|   |-- shap_summary.png
|   |-- shap_bar.png
|   `-- shap_force_plot.html
|-- src/
`-- requirements.txt
```

## Dataset

- Source: scikit-learn `load_breast_cancer()` (exported to CSV in this project)
- Samples: 569
- Features: 30 numeric predictors
- Target: 1 output column (`target`)
- Total columns in CSV: 31 (30 features + target)
- Class distribution:
  - `target = 1`: 357 (benign)
  - `target = 0`: 212 (malignant)

## Workflow

1. `notebooks/data_loading.ipynb`
   - Loads dataset from sklearn
   - Converts to DataFrame and saves `data/breast_cancer.csv`

2. `notebooks/step2_eda.ipynb`
   - Basic data inspection (`shape`, `info`, `describe`)
   - Class distribution plot
   - Correlation heatmap
   - Confirms no missing values

3. `notebooks/step3_preprocessing.ipynb`
   - Train/test split (80/20)
   - Feature scaling with `StandardScaler` fitted on training data only to prevent leakage

4. `notebooks/step4_model_training.ipynb`
   - Trains `RandomForestClassifier`, `SVC(probability=True)`, and `XGBClassifier`
   - Evaluates accuracy, classification reports, and 5-fold cross-validation for all models

5. `notebooks/step5_shap_interpretability.ipynb`
   - Trains XGBoost for explainability analysis
   - Computes SHAP values
   - Exports summary, bar, and force-plot reports

## Model Results

### Test-Set Performance (80/20 split)

| Model | Accuracy | Macro Avg (P/R/F1) | Weighted Avg (P/R/F1) |
|------|----------|--------------------|------------------------|
| Random Forest | 96.49% | 0.97 / 0.96 / 0.96 | 0.97 / 0.96 / 0.96 |
| SVM | 98.25% | 0.99 / 0.98 / 0.98 | 0.98 / 0.98 / 0.98 |
| XGBoost | 95.61% | 0.96 / 0.95 / 0.95 | 0.96 / 0.96 / 0.96 |

### Cross-Validation Results (5-Fold)

| Model | Mean Accuracy | Std Dev |
|------|---------------|---------|
| Random Forest | 95.61% | +/- 2.28% |
| SVM | 97.36% | +/- 1.47% |
| XGBoost | 97.36% | +/- 1.47% |

SVM and XGBoost achieved the highest mean accuracy (97.36%) with low variance (+/- 1.47%), confirming stable generalization across all folds. Random Forest maintained solid performance with slightly higher variance.

## Explainability Results (SHAP)

Generated artifacts in `reports/`:

- `shap_summary.png`: SHAP beeswarm summary (global feature impact + directionality)
- `shap_bar.png`: mean absolute SHAP feature importance ranking
- `shap_force_plot.html`: interactive force plot for individual prediction explanation

Top features by SHAP importance:

1. `mean concave points`
2. `worst area`
3. `worst concave points`
4. `worst texture`
5. `area error`

### SHAP Summary Plot

![SHAP Summary](reports/shap_summary.png)

### SHAP Feature Importance Bar Plot

![SHAP Bar](reports/shap_bar.png)

To view local explanation interactively, open `reports/shap_force_plot.html` in a browser.

## Environment Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## How To Run

Run notebooks in order:

1. `notebooks/data_loading.ipynb`
2. `notebooks/step2_eda.ipynb`
3. `notebooks/step3_preprocessing.ipynb`
4. `notebooks/step4_model_training.ipynb`
5. `notebooks/step5_shap_interpretability.ipynb`

## Notes

- Results include both holdout test-set evaluation and 5-fold cross-validation for robust comparison.
- This is not a clinical diagnostic system.

## Future Improvements

- Hyperparameter tuning with GridSearchCV
- ROC-AUC and PR-AUC curve comparison across models
- Model persistence with joblib for inference deployment
- Inference script in `src/` for standalone prediction
