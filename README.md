# README

## Introduction
This repository contains a Python script for data analysis and machine learning tasks using the pandas, numpy, seaborn, matplotlib, and scikit-learn libraries. The script loads a dataset, performs data preprocessing, explores data insights, handles outliers, and trains a linear regression model.

## Dependencies
Make sure you have the following dependencies installed:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

You can install these dependencies using pip:
```
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Usage
1. Clone the repository:
```
git clone <repository_url>
```

2. Navigate to the directory containing the script.

3. Run the script using Python:
```
python data_analysis_and_ml.py
```

## Description
The script performs the following tasks:

1. Import required libraries:
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib
```

2. Load the dataset:
```python
tm = pd.read_csv("E:\\archive (6).zip")
```

3. Explore the dataset:
```python
tm.head()
tm.describe()
tm.isnull().sum()
```

4. Visualize correlation matrix:
```python
correlation_matrix = tm.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
```

5. Data preprocessing:
```python
tm.drop('Loud Cover', axis=1, inplace=True)
tm['Formatted Date'] = pd.to_datetime(tm['Formatted Date'], utc=True)
tm['year'] = tm['Formatted Date'].dt.year
tm['month'] = tm['Formatted Date'].dt.month
tm['day'] = tm['Formatted Date'].dt.day
tm['weekday'] = tm['Formatted Date'].dt.weekday
tm['precip_type'] = tm['Precip Type'].map({'rain': 0, 'snow': 1}).fillna(0)
lbl_encoder = preprocessing.LabelEncoder()
tm['summary'] = lbl_encoder.fit_transform(tm['Summary'])
tm['Daily Summary'] = lbl_encoder.fit_transform(tm['Daily Summary'])
del tm['Summary']
del tm['Formatted Date']
del tm['Precip Type']
```

6. Visualize outliers:
```python
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k, v in tm.items():
    sns.boxplot(y=k, data=tm, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()
```

7. Train a linear regression model:
```python
x = tm.drop('Temperature (C)', axis=1)
y = tm['Temperature (C)']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
model_1 = LinearRegression()
model_1.fit(x_train, y_train)
y_pred = model_1.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("R2 Linear Regression Score: {:1%}".format(r2))
plt.scatter(y_test, y_pred, c='r')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()
```
