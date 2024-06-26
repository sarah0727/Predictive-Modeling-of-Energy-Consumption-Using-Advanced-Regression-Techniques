# Advanced Regression Project Report: Energy Efficiency Prediction

## Introduction

This project aims to develop an advanced regression model to predict the heating and cooling load requirements of buildings based on various physical attributes. The dataset used for this project is the UCI Energy Efficiency Dataset, which contains eight features related to building characteristics and two target variables: Heating Load (HL) and Cooling Load (CL).

## Dataset

### Description

The dataset used for this project is the Energy Efficiency Dataset, available from the UCI Machine Learning Repository. This dataset consists of 768 samples and 8 features that describe the physical attributes of buildings. The target variables are Heating Load (HL) and Cooling Load (CL).


### Features

- `X1`: Relative Compactness
- `X2`: Surface Area
- `X3`: Wall Area
- `X4`: Roof Area
- `X5`: Overall Height
- `X6`: Orientation
- `X7`: Glazing Area
- `X8`: Glazing Area Distribution

### Target Variables

- `Y1`: Heating Load (HL)
- `Y2`: Cooling Load (CL)

### Citation

The dataset is cited as follows:

> Tsanas,Athanasios and Xifara,Angeliki. (2012). Energy Efficiency. UCI Machine Learning Repository. https://doi.org/10.24432/C51307.
## Data Preprocessing

### Data Cleaning

The dataset was inspected for missing values and inconsistencies. No missing values were found.

### Feature Scaling

Feature scaling was performed to normalize the features. StandardScaler from scikit-learn was used to scale the features to have zero mean and unit variance. This step is crucial for algorithms like Gradient Boosting and Support Vector Machines, which are sensitive to the scale of the input data.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Model Development

### Algorithms Used

1. **Gradient Boosting Regressor (GBR)**: A powerful ensemble learning technique that builds models sequentially and reduces the bias of the combined model.

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(X_train, y_train)
```

### Model Training

The dataset was split into training and testing sets with an 80-20 split. The model was trained on the training set, and hyperparameters were tuned using cross-validation to find the optimal model configuration.

### Model Evaluation

The model was evaluated using metrics such as Mean Absolute Error (MAE) and R-squared (R²). These metrics provide insight into the model's accuracy and the proportion of variance explained by the model.

```python
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

## Results

### Model Performance

- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions.
- **R-squared (R²)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

### Visualizations

1. **Feature Importances**: Visualizing feature importances helps understand which features contribute most to the model's predictions.

```python
import matplotlib.pyplot as plt

feature_importances = model.feature_importances_
plt.barh(range(len(feature_importances)), feature_importances)
plt.yticks(range(len(feature_importances)), ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()
```

2. **Actual vs. Predicted Values**: Comparing the actual and predicted values for Heating Load and Cooling Load provides a visual understanding of the model's performance.

```python
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
```

### SHAP Analysis

SHAP (SHapley Additive exPlanations) was used to interpret the model's predictions by explaining the contribution of each feature to the prediction.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Dependence plot for 'X1' (Relative Compactness)
shap.dependence_plot('X1', shap_values, X_test)
```

## Conclusion

The advanced regression model developed in this project effectively predicts buildings' heating and cooling load requirements based on their physical attributes. Gradient Boosting Regressor and SHAP analysis provided accurate predictions and valuable insights into feature importance and model interpretability.

### Future Work

Future improvements could include:

- Experimenting with other advanced regression algorithms.
- Implementing hyperparameter tuning using techniques such as GridSearchCV or RandomizedSearchCV.
- Exploring the impact of additional features or external datasets to enhance model performance.

### References

- UCI Machine Learning Repository: Energy Efficiency Data Set. Available: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency
