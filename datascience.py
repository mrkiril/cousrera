import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import os
import csv
import time
import statsmodels.api as sm

from collections import namedtuple
from os import listdir
from os.path import isfile, join
from sys import argv
from pprint import pprint as pp

from helpers import diki

path = os.path.dirname(os.path.abspath(__file__))
path_to_data_folder = join(path, 'data')

milk = pd.read_csv(join(path_to_data_folder, 'milk.csv'), ';', index_col=['month'], parse_dates=['month'], dayfirst=True)
milk['daily'] = [v / k.days_in_month for k, v in milk['milk'].items()]

print('SUM: ', sum(milk.daily))

# milk.plot()
# milk.daily.plot()
# plt.show()


# Diki Fuller
# Якщо р близько 0 отже ряд стаціонарний
# If |p| близько 0 отже ряд нестаціонарний

print('DF: ', diki(milk['milk']))
print('DF by day: ', diki(milk.daily))




milk.daily_diff12 = milk.daily - milk.daily.shift(12)
milk.daily_diff12.dropna(inplace=True)

milk.daily_diff12_diff = milk.daily_diff12 - milk.daily_diff12.shift(1)
milk.daily_diff12_diff.dropna(inplace=True)
# milk.daily_diff1 = milk.daily - milk.daily.shift(1)
# milk.daily_diff1.dropna(inplace=True)



milk.daily_diff12_diff.plot(title="Diff_diff12, diki: " + str(diki(milk.daily_diff12_diff)))
plt.show()

# milk.daily_diff12 = milk.daily - milk.daily.shift(12)
# milk.daily_diff12.dropna(inplace=True)
# milk.daily_diff12.plot(title="Diff12, diki: " + str(diki(milk.daily_diff12)))
# plt.show()



sm.graphics.tsa.plot_acf(milk.daily_diff12_diff.values.squeeze(), lags=50)
plt.show()


sm.graphics.tsa.plot_pacf(milk.daily_diff12_diff.values.squeeze(), lags=50)
plt.show()

