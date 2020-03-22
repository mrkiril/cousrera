import statsmodels.api as sm
import numpy as np

from scipy import stats


def diki(series, acc=5):
    return round(sm.tsa.stattools.adfuller(series)[1], acc)


def student(series, popmean=0):
    return stats.ttest_1samp(series, popmean)[1]


def invboxcox(y, lmbda):
    if lmbda == 0:
        return np.exp(y)
    else:
        return np.exp(np.log(lmbda*y+1)/lmbda)
