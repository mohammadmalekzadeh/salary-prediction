### importing library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import joblib

### read and split data
data = pd.read_csv('../data/processed/train_encoding_data.csv')
X = data.drop(columns='Annual_salary')
y = data['Annual_salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_test.to_csv('../data/test_data/X_test.csv', index=False)
y_test.to_csv('../data/test_data/y_test.csv', index=False)

### definition models list
models_list = [('lr', LinearRegression()),
    ('dt', DecisionTreeRegressor(max_depth=10, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, activation='relu', solver='adam', n_iter_no_change=True, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)),
    ('et', ExtraTreesRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42))]

### train and save models
for name, model in models_list:
    model.fit(X_train, y_train)
    joblib.dump(model, f'../models/salary.{name}_model')