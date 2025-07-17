### importing library
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
import joblib

### read test data and model
test_data = pd.read_csv('../data/processed/test_encoding_data.csv')
gbr_model = joblib.load('../models/salary.gb_model')
salary_true = pd.read_csv('../data/true_values/y_true.csv')

### prediction
salary_prediction = gbr_model.predict(test_data)

### prediction score calculation
mse = mse(salary_true, salary_prediction)
mae = mae(salary_true, salary_prediction)
r2 = r2(salary_true, salary_prediction)

### save result prediction
salary_prediction = pd.DataFrame({'Salary Predicton': salary_prediction})
salary_prediction.to_csv('../results/salary_prediction.csv', index=False)
prediction_score = pd.DataFrame([{'mae': mae, 'mse': mse, 'r2': r2}])
prediction_score.to_csv('../results/prediction_score.csv', index=False)

# output:
# Mean Absolute Error Score: 14380.354802006897
# Mean Squared Error Score: 304370850.16651917
# R2 Score: 0.8836355758328941