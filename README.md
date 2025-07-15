# Salary Prediction Project 💰

## 🔍 Introduction
This project predicts individual salary using machine learning techniques applied on a structured dataset of personal and professional attributes (e.g., years of experience, location).  
Ideal for: HR analytics, job-market insights, compensation benchmarking.

## 🧰 Features & Data
- **Feature set**: job title, experience years, industry, location, etc.  
- **Data preprocessing**: missing value handling, encoding, normalization.  
- **Model**: regression algorithm (e.g., Random Forest / Neural Networks / SVR), hyperparameter tuning.  
- **Outputs**: salary estimates, RMSE & MAE evaluation

## 🚀 Technologies Used
```bash
    - Jupyter Notebook
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    - scikit-learn

```

## 🖇 Prerequisites
```python
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - scikit-learn
    - datetime
    - joblib

```

## ⚙️ Installation
```bash
    - git clone https://github.com/mohammadmalekzadeh/salary-prediction.git
    - cd salary-prediction
    - python3 -m venv venv
    - source venv/bin/activate
    - pip install -r requirements.txt
```


## 📁 Project Structure
```
salary-prediction/
├── data/
│   ├── raw/
│   ├── true_values/
│   └── processed/
├── src/
│   ├── processing.py
│   ├── model.py
│   ├── predict.py
│   └── evaluate.py
├── models/
├── notebooks/
├── results/
├── rports/
│   └── chart/
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🧠 Usage

### 1. Preprocessing  
```bash
python src/processing.py --data data/raw/train_data.csv --output data/processed/
```

### 2. Training  
```bash
python src/model.py \
  --train data/processed/train_encoding_data.csv \
  --val data/true_values/y_true.csv \
  --model_output models/salary.model
```

### 3. Prediction  
```bash
python src/predict.py \
  --model models/salary.model \
  --input data/predict/new_samples.csv \
  --output results/predictions.csv
```

### 4. Evaluation  
```bash
python src/evaluate.py \
  --predictions results/predictions.csv \
  --ground_truth data/true_values/y_true.csv
```


## 📜 License
This project is licensed under the [MIT License](LICENSE).

## 📬 Contact
Maintained by **Mohammad Malekzadeh**.  
Questions? Issues? Feature requests? Just open an issue or reach out via GitHub!