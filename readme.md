# Polynomial Regression Modeling (BT2024148)

This repository contains the code, data, and predictions for a polynomial regression assignment. The goal is to predict a continuous target variable `y` across two distinct synthetic datasets (`var1` and `var2`).

## Files Included
* `train_and_predict.py`: The main execution script. It performs feature subset selection, polynomial degree sweeps, and evaluates OLS vs. Regularized models (Ridge/Lasso).
* `BT2024148_train_var1.csv` & `BT2024148_train_var2.csv`: The personalized training datasets.
* `BT2024148_pred_var1.csv` & `BT2024148_pred_var2.csv`: The final model predictions on the unseen test data.
* `figures/`: Contains generated plots for error sweeping and model comparisons.
* `README.md`: Project documentation.

## Final Models Selected
1. **Var1:** Lasso Regression (Degree 5)
   * All 6 features used. 
   * Lasso regularization reduced 461 possible polynomial terms down to 145 active terms, mitigating overfitting.
   * Validation R²: 0.966
2. **Var2:** Ordinary Least Squares (Degree 8)
   * All 3 features (`x1`, `x2`, `x3`) used.
   * Validation R²: 0.994

## How to Run
Ensure you have the dependencies installed:
`pip install numpy pandas scikit-learn matplotlib`

Run the main script:
`python train_predict.py`