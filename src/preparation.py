### importing library

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

### read .csv file

df = pd.read_csv('../data/raw_data.csv')

### columns_name changed

df.columns = ['Id', 'Fullname', 'Job_title', 'Department', 'Business_unit', 'Gender', 'Ethnicity', 'Age', 'Hiring_date', 'Annual_salary', 'Award', 'Country', 'City']

### definition tha maps

job_title_map = {
    'تحلیلگر': 'Analyst',
    'مدیر': 'Manager',
    'کارگردان': 'Director',
    'معاونت': 'Assistant',
    'مهندس فنی': 'Technical Engineer',
    'مهندس سازمانی': 'Enterprise Engineer',
    'مدیر سیستم': 'System Manager',
    'مهندس توسعه': 'Development Engineer',
    'مهندس کنترل': 'Control Engineer',
    'مهندس زیرساخت ابری': 'Cloud Infrastructure Engineer',
    'مدیر شبکه': 'Network Manager',
    'شریک تجاری': 'Business Partner',
    'مهندس آزمون': 'Test Engineer',
    'مهندس شبکه': 'Network Engineer',
    'مدیر توسعه': 'Development Manager',
    'مهندس کیفیت': 'Quality Engineer',
    'مهندس اتوماسیون': 'Automation Engineer'
}

department_map = {
    'حسابداری': 'Accounting',
    'آی‌تی': 'IT',
    'مهندسی': 'Engineering',
    'بازاریابی': 'Marketing',
    'منابع انسانی': 'Human Resources'
}

unit_map = {
    'محصولات تخصصی': 'Specialized Products',
    'شرکت‌های بزرگ': 'Large Companies',
    'تحقیق و توسعه': 'Research and Development',
    'تولید': 'Production'
}

ethnicity_map = {
    'آسیایی': 'Asian',
    'لاتین': 'Latin',
    'سیاه پوست': 'Black'
}

country_map = {
    'آمریکا': 'USA',
    'چین': 'China',
    'برزیل': 'Brazil'
}

city_map = {
    'میامی': 'Miami',
    'آستین': 'Austin',
    'سیاتل': 'Seattle',
    'شیکاگو': 'Chicago',
    'فینیکس': 'Phoenix',
    'کلمبوس': 'Columbus',
    'چونگ‌کینگ': 'Chongqing',
    'چنگدو': 'Chengdu',
    'پکن': 'Beijing',
    'شانگهای': 'Shanghai',
    'ریو دو ژانیرو': 'Rio de Janeiro',
    'مانائوس': 'Manaus',
    'سائو پائولو': 'São Paulo'
}

### applay mapping

df.Job_title = df.Job_title.map(job_title_map)
df.Department = df.Department.map(department_map)
df.Business_unit = df.Business_unit.map(unit_map)
df.Gender = df.Gender.map({'مرد': 'Male', 'زن': 'Female'})
df.Ethnicity = df.Ethnicity.map(ethnicity_map)
df.Country = df.Country.map(country_map)
df.City = df.City.map(city_map)

### split data and save them

train_data, test_data = train_test_split(df, test_size=0.2, random_state=41)

test_data.reset_index(inplace=True)
test_data.drop(columns='index')
test_data.Annual_salary.to_csv('../data/y_true.csv', index=False)
test_data.drop(columns='Annual_salary').to_csv('../data/test_dara.csv', index=False)

train_data.reset_index(inplace=True)
train_data.drop(columns='index')
train_data.to_csv('../data/train_data.csv', index=False)