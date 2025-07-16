### importing library
import numpy as np
import pandas as pd

### read reports
reports = pd.read_csv('../reports/models/models_performance_report.csv')

### sorting models by score
best_model_mae = reports.groupby('model')['mae'].sum().sort_values(ascending=True).idxmax()
best_model_mse = reports.groupby('model')['mse'].sum().sort_values(ascending=True).idxmax()
best_model_r2 = reports.groupby('model')['r2'].sum().sort_values(ascending=False).idxmax()

### result
print(best_model_mae and best_model_mse and best_model_r2)
# output => "GradientBoostingRegressor"
# Mean Absolute Error Score: 14202.455179618468
# Mean Squared Error Score: 302895035.97748834
# R2 Score: 0.8912808204649216
