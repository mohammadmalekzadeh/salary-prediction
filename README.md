# 💰 Salary Prediction Project

## 🔍 Introduction
This project predicts individual salary using machine learning techniques applied on a structured dataset of personal and professional attributes (e.g., years of experience, location).  
Ideal for: HR analytics, job-market insights, compensation benchmarking.

## 🧰 Features & Data
- **Feature set**: job title, experience years, industry, location, etc.  
- **Data preprocessing**: missing value handling, encoding, normalization.  
- **Model**: regression algorithm (e.g., Random Forest / Neural Networks / SVR), hyperparameter tuning.  
- **Outputs**: salary estimates, RMSE & MAE evaluation

## 📌 Result Summary
For this project and salary prediction, **```GradientBoostingRegressor M.L. Model```** had the best accuracy
- **Mean Absolute Error Score**: ```14202.455179618468```
- **Mean Squared Error Score**: ```302895035.97748834```
- **R2 Score**: ```0.8912808204649216```

## 🚀 Technologies Used
```bash
    - Jupyter Notebook
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    - scikit-learn

```


## 📁 Project Structure
```
salary-prediction/
├── data/
│   ├── raw/
│   ├── true_values/
│   ├── test_data/
│   └── processed/
├── src/
│   ├── processing.py
│   ├── model.py
│   ├── predict.py
│   ├── best_model_selcting.py
│   └── validation.py
├── models/
├── notebooks/
├── results/
├── rports/
│   └── chart/
│   └── models/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```


## 📜 License
This project is licensed under the [MIT License](LICENSE).

## 📬 Contact
Maintained by **Mohammad Malekzadeh**.  
Questions? Issues? Feature requests? Just open an issue or reach out via GitHub!
