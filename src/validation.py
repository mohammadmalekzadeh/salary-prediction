### importing library
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import joblib
import os

### read test dataframe and models
X_test = pd.read_csv('../data/test_data/X_test.csv')
y_test = pd.read_csv('../data/test_data/y_test.csv')
models_list = []
models_name_list = {'lr': 'Linear Regression',
    'dt': 'Decision Tree Regressor',
    'rf': 'RandomForestRegressor',
    'mlp': 'MLP Regressor',
    'gb': 'GradientBoostingRegressor',
    'et': 'ExtraTreesRegressor',
    'xgb': 'XGBRegressor'}
for model in os.listdir('../models'):
    models_list.append([models_name_list[model.split('.')[1].split('_')[0]], joblib.load(f'../models/{model}')])

### definition validation report and models validation
models_performance_report = pd.DataFrame(columns=['model', 'mae', 'mse', 'r2'])
for name, model in models_list:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    models_performance_report = pd.concat([models_performance_report, pd.DataFrame([{'model': name, 'mae': mae, 'mse': mse, 'r2': r2}])], ignore_index=True)

### save models performance report
models_performance_report.to_csv('../reports/models/models_performance_report.csv', index=False)