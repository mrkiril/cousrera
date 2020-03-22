import csv
import io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
import time
import warnings

from collections import namedtuple
from datetime import datetime
from dateutil.relativedelta import relativedelta
from itertools import product
from os import listdir
from os.path import isfile, join
from pprint import pprint as pp
from scipy import stats
from sys import argv

from helpers import diki, student, invboxcox

#  1.   Критерій Дікі-Фуллера має бути влизьким до 0
#  2.   STL-decompose - щоб зрозуміти з яких частин складається ряд
#  3.   Перетворення BoxCox для стабілізації дисперсії
#  4.   Диференціювання ряда (сезонне)
#  5.   STL-decompose - для оцінки ряда після сезонного диференціювання
#  6.   Диференціювання ряда (звичайне)
#  7.   STL-decompose - для оцінки ряда після звичайного диференціювання
#  8.   Підбір параметрів p, q, P, Q з Авто і ЧастковоАвтокореляційних функцій
#  9.   Підбір найкращої моделі шляхом перебору всіх комбінацій параметрів з Найменших коеф. Акаікі
#  10.  Візуальний аналіз залишків моделі
#  11.  Побудова опису данних моделлю
#  12.  Predict

path = os.path.dirname(os.path.abspath(__file__))
path_to_data_folder = join(path, 'data')


#  Step 1.
#  Diki: 0.99
salary = pd.read_csv(join(path_to_data_folder, 'russian_salary.csv'), ';', index_col=['month'], parse_dates=['month'], dayfirst=True)
# salary.salary.plot(title="Diki: " + str(diki(salary.salary)))
print('Diki salary original: ', diki(salary.salary))


#  Step 2.
#  STL-decompose
# sm.tsa.seasonal_decompose(salary.salary).plot()
# plt.show()


#  Step 3.
#  BoxCox
#  After BoxCox - Diki: 0.6969
salary['salary_box'], lmbda = stats.boxcox(salary.salary)
print('Diki salary box_cox: ', diki(salary.salary_box))
# salary.salary_box.plot(title="Diki box_cox: " + str(diki(salary.salary_box)))
# plt.show()


#  Step 4.
#  Диференціювання ряда
#  Diki: 0.0147
salary['salary_box_diff12'] = salary.salary_box - salary.salary_box.shift(12)
salary.salary_box_diff12.dropna(inplace=True)
# salary.salary_box_diff12.plot(title="salary_box_diff12, diki: " + str(diki(salary.salary_box_diff12)))
# plt.show()
print('Diki salary_box_diff12: ', diki(salary.salary_box_diff12))


#  Step 5.
#  STL для оцінки ряда
#  Trend доволі поганий
# sm.tsa.seasonal_decompose(salary.salary_box_diff12).plot()
# plt.show()


#  Step 6.
#  Диференціювання ряда
#  Diki: 5e*10^-8
#  Гіпотеза нестаціонарності впевнено відхиляється
salary['salary_box_diff1'] = salary.salary_box_diff12 - salary.salary_box_diff12.shift(1)
salary.salary_box_diff1.dropna(inplace=True)
# salary.salary_box_diff1.plot(title="salary_box_diff1, diki: " + str(diki(salary.salary_box_diff1)))
# plt.show()
print('Diki salary_box_diff1: ', diki(salary.salary_box_diff1, acc=10))


#  Step 7.
#  STL для оцінки ряда
# sm.tsa.seasonal_decompose(salary.salary_box_diff1).plot()
# plt.show()


#  Step 8.
#  Побудова автокореляційної і частково автокореляційної функції
#  Для визначення параметрів p, q, P, Q
plt.figure(figsize=(15, 8))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(salary.salary_box_diff1.values.squeeze(), lags=50, ax=ax)

ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(salary.salary_box_diff1.values.squeeze(), lags=50, ax=ax)
plt.show()

Qs = range(0, 2)  # 0, 0
qs = range(0, 1)  # 0, 1
Ps = range(0, 2)  # 0, 4
ps = range(0, 1)  # 0, 1
d = 1
D = 1


#  Step 9.
#  Увага в функцію SARIMAX ми відправляємо ряд після перетворення Бокса-Кокса
#  Параметри найкращої моделі - (1, 0, 1, 0)  akaik - 0.340521
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)

results = []
best_aic = float('inf')
warnings.filterwarnings('ignore')
for param in parameters_list:
    # try except нужен, потому что на некоторых наборах параметров модель не обучается
    try:
        model = sm.tsa.statespace.SARIMAX(
            salary.salary_box,
            order=(param[0], d, param[1]),
            seasonal_order=(param[2], D, param[3], 12)
        ).fit(disp=-1)

    # выводим параметры, на которых модель не обучается и переходим к следующему набору
    except ValueError:
        print('wrong parameters:', param)
        continue

    aic = model.aic
    # сохраняем лучшую модель, aic, параметры
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])

warnings.filterwarnings('default')


results_table = pd.DataFrame(results)
results_table.columns = ['parameters', 'aic']
print( results_table.sort_values(by='aic', ascending=[True]).head() )


#  Step 10.
#  Analyse Residuals of the model
#  Residuals - less the border lvl
#  t-criterio Student: 0.091
#  Diki: 3.4588e-06
plt.figure(figsize=(15, 8))
plt.subplot(211)
plt.ylabel('Residuals')
best_model.resid[13:].plot()
ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=50, ax=ax)
print('= = = = = = = = = = = = = = = = = ')
print('= = = = = = = = = = = = = = = = = ')
print("Residuals Student Q: ", student(best_model.resid[13:]) )
print("Residuals Diki:      ", diki(best_model.resid[13:], acc=10) )


#  Step 11.
#  Draw model by data
#  Малюємо побудовану модель поверх огигінального графіка
salary['model'] = invboxcox(best_model.fittedvalues, lmbda)
plt.figure(figsize=(15, 8))
salary.salary.plot()
salary.model.plot(color='r')
plt.ylabel('Salary lvl')
plt.show()


#  Step 12.
#  Predict data
predict_month = 36
salary2 = salary[['salary']]
date_list = [
    datetime.strptime('2016-09-01', '%Y-%m-%d') + relativedelta(months=x)
    for x in range(0, predict_month)
]
future = pd.DataFrame(index=date_list, columns=salary2.columns)
salary2 = pd.concat([salary2, future])
salary2['forecast'] = invboxcox(best_model.predict(start=len(salary), end=len(salary)+predict_month), lmbda)

plt.figure(figsize=(15, 7))
salary2.salary.plot()
salary2.forecast.plot(color='r')
plt.ylabel('Russian Salary')
plt.plot()
plt.show()
