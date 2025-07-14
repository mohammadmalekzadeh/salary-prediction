### importing library
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

### read datafram
train_data = pd.read_csv('../data/train_data.csv')
test_data = pd.read_csv('../data/test_dara.csv')
train_data.head()
train_data.isnull().sum()

### dropping extra columns
extra_col = ['Id', 'Fullname']
train_data.drop(columns=extra_col, inplace=True)
test_data.drop(columns=extra_col, inplace=True)

### add new feature
today = datetime.date.today().year
train_data['Hiring_year'] = pd.to_datetime(train_data['Hiring_date']).dt.year
train_data['Work_duration'] = today - train_data['Hiring_year']
train_data.drop(columns='Hiring_date', inplace=True)
test_data['Hiring_year'] = pd.to_datetime(test_data['Hiring_date']).dt.year
test_data['Work_duration'] = today - test_data['Hiring_year']
test_data.drop(columns='Hiring_date', inplace=True)
mean_salary_by_country = train_data.groupby('Country')['Annual_salary'].mean().round(2).to_dict()
mean_salary_by_city = train_data.groupby('City')['Annual_salary'].mean().round(2).to_dict()
train_data['mean_salary_by_country'] = train_data['Country'].map(mean_salary_by_country)
train_data['mean_salary_by_city'] = train_data['City'].map(mean_salary_by_city)
test_data['mean_salary_by_country'] = test_data['Country'].map(mean_salary_by_country)
test_data['mean_salary_by_city'] = test_data['City'].map(mean_salary_by_city)
mean_salary_by_job_title = train_data.groupby('Job_title')['Annual_salary'].mean().round(2).to_dict()
train_data['mean_salary_by_job_title'] = train_data['Job_title'].map(mean_salary_by_job_title)
test_data['mean_salary_by_job_title'] = test_data['Job_title'].map(mean_salary_by_job_title)

### normalization
ss = StandardScaler()
train_data['Award'] = ss.fit_transform(train_data[['Award']])
test_data['Award'] = ss.transform(test_data[['Award']])

### save the clean data before label encoding
train_data.to_csv('../data/train_clean_data.csv', index=False)
test_data.to_csv('../data/test_clean_data.csv', index=False)

### label Encoding
le = LabelEncoder()
le_list = ['Job_title', 'Department', 'Business_unit', 'Gender', 'Ethnicity', 'Country', 'City']
for col in le_list:
    train_data[col] = le.fit_transform(train_data[[col]])
    test_data[col] = le.transform(test_data[[col]])

### save the clean and encoding data
train_data.to_csv('../data/train_encoding_data.csv', index=False)
test_data.to_csv('../data/test_encoding_data.csv', index=False)