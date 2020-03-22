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


path = os.path.dirname(os.path.abspath(__file__))
path_to_data_folder = join(path, 'data')

wine = pd.read_csv(join(path_to_data_folder, 'monthly-australian-wine-sales.csv'), ',', index_col=['month'], parse_dates=['month'], dayfirst=True)
wine.sales = wine.sales * 1000

wine.sales.plot(title="Diki: " + str(diki(wine.sales)))
print('Diki wine: ', diki(wine.sales))


# sm.tsa.seasonal_decompose(wine.sales).plot()
# plt.show()


wine['sales_box'], lmbda = stats.boxcox(wine.sales)
wine.sales_box.plot(title="Diki: " + str(diki(wine.sales_box)))
# plt.show()

plt.ylabel(u'Transformed wine sales')
print("Оптимальный параметр преобразования Бокса-Кокса: %f" % lmbda)
print("Критерий Дики-Фуллера: p=%f" % diki(wine.sales_box))




wine['sales_box_diff12'] = wine.sales_box - wine.sales_box.shift(12)
wine.sales_box_diff12.dropna(inplace=True)
wine.sales_box_diff12.plot(title="sales_box_diff12, diki: "+ str(diki(wine.sales_box_diff12)))
sm.tsa.seasonal_decompose(wine.sales_box_diff12).plot()
# plt.show()



wine['sales_box_diff1'] = wine.sales_box_diff12 - wine.sales_box_diff12.shift(1)
wine.sales_box_diff1.dropna(inplace=True)
# wine.sales_box_diff1.plot(title="sales_box_diff1, diki: " + str(diki(wine.sales_box_diff1, 10)))
sm.tsa.seasonal_decompose(wine.sales_box_diff1).plot()
# plt.show()


plt.figure(figsize=(15, 8))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(wine.sales_box_diff1.values.squeeze(), lags=48, ax=ax)
# pylab.show()

ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(wine.sales_box_diff1.values.squeeze(), lags=48, ax=ax)
plt.show()

Qs = range(0, 2)
qs = range(0, 2)  # 0, 3
Ps = range(0, 1)  # 0, 2
ps = range(0, 3)  # 0, 5
d = 1
D = 1

parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)

results = []
best_aic = float('inf')
warnings.filterwarnings('ignore')
for param in parameters_list:
    # try except нужен, потому что на некоторых наборах параметров модель не обучается
    try:
        model = sm.tsa.statespace.SARIMAX(
            wine.sales_box,
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
# pp(results)


results_table = pd.DataFrame(results)
results_table.columns = ['parameters', 'aic']
print(
    results_table.sort_values(by='aic', ascending=[True]).head()
)


plt.figure(figsize=(15, 8))
plt.subplot(211)
plt.ylabel('Residuals')
best_model.resid[13:].plot()

ax = plt.subplot(212)
sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48, ax=ax)
print('= = = = = = = = = = = = = = = = = ')
print('= = = = = = = = = = = = = = = = = ')
print("Student Q: ", student(best_model.resid[13:]) )
print('Diki: ', diki(best_model.resid[13:], acc=10) )


#  Малюємо побудовану модель поверх огигінального графіка
wine['model'] = invboxcox(best_model.fittedvalues, lmbda)
plt.figure(figsize=(15, 8))
wine.sales.plot()
wine.model[13:].plot(color='r')
plt.ylabel('Wine Sales')
plt.show()



#  Predict
#
wine2 = wine[['sales']]
date_list = [
    datetime.strptime('1994-09-01', '%Y-%m-%d') + relativedelta(months=x)
    for x in range(0, 36)
]
future = pd.DataFrame(index=date_list, columns=wine2.columns)
wine2 = pd.concat([wine2, future])
wine2['forecast'] = invboxcox(best_model.predict(start=176, end=211), lmbda)

plt.figure(figsize=(15, 7))
wine2.sales.plot()
wine2.forecast.plot(color='r')
plt.ylabel('Wine Sales')
plt.plot()
plt.show()
