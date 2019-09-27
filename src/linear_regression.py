'''@file: linear_regression.py
@content: do the prediction for three weeks on stock data
----------------------------------------------------------
@author: Tianyi Wang
@modified time: 09/24/2019
'''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


def plotting_single_file(loc, plot_list):
    data = pd.read_csv(loc)
    data_length = len(data[plot_list[0]])
    x = np.arange(data_length)[-365:]
    y = data[plot_list[0]][-365:]
    x = x.reshape(-1, 1)

    reg = linear_model.LinearRegression()
    reg.fit(x, y)

    print(f'regression coef: {reg.coef_} and\
    regression intercept: {reg.intercept_}')
    plt.title(f'time vs {plot_list[0]}')

    x_min = x.min()
    x_max = x.max()

    plt.plot([x_min, x_max],
             [reg.coef_[0] * x_min + reg.intercept_,
              reg.coef_[0] * x_max + reg.intercept_], c='g',
             label='predicted')
    plt.xlabel('time')
    plt.ylabel(f'{plot_list[0]}')
    plt.scatter(x, y, label='actual', c='r', s=2)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plotting_single_file('../raw/AAPL.csv', ['Adj Close'])
