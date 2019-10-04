'''@file: linear_regression.py
@content: do the prediction for three weeks on stock data
----------------------------------------------------------
@author: Tianyi Wang
@modified time: 10/02/2019
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


def k_neighbor(x, y, x_pre, y_pre):
    from sklearn import neighbors
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.1)
    standard = MinMaxScaler(feature_range=(0, 1))
    x_train = standard.fit_transform(x_train)
    knn_model = neighbors.KNeighborsRegressor()
    parameters = {'n_neighbors': np.arange(2, 40).tolist()}
    reg = GridSearchCV(knn_model, parameters, cv=10)
    reg.fit(x_train, y_train)
    ax.plot(pd.DataFrame(reg.predict(standard.fit_transform(x)),
                         index=y.index), c='k',
            label='train_set_predict_value')
    return pd.DataFrame(reg.predict(standard.fit_transform(x_pre)),
                        index=y_pre.index)


def get_data(loc, predict_object, length, pre_length):
    data = pd.read_csv(loc, index_col=['Date'])
    return data.drop([predict_object], axis=1)[-length -
                                               pre_length:-pre_length],\
        data[predict_object][-length - pre_length:-pre_length],\
        data.drop([predict_object], axis=1)[-pre_length:],\
        data[predict_object][-pre_length:]


if __name__ == '__main__':
    import argparse
    import matplotlib.ticker as ticker

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--location',
                        help='the location of your source csv file')
    parser.add_argument('-t', '--target',
                        help='the target which you want to do prediction on')
    parser.add_argument('-ptl', '--practice_length',
                        help='the length of data which you want to use in\
                        your practice set for your prediction model')
    parser.add_argument('-pdl', '--predict_length',
                        help='the length of data which you want to use in\
                        your predict set')
    parser.add_argument('-m', '--method',
                        help='scatter or plot')
    args = parser.parse_args()
    fig, ax = plt.subplots(1, 1, figsize=(30, 10))
    x, y, x_pre, y_pre = get_data(args.location, args.target,
                                  int(args.practice_length),
                                  int(args.predict_length))
    y_pre_ = k_neighbor(x, y, x_pre, y_pre)

    error = 0
    for i in range(len(y_pre)):
        error += np.abs(y_pre.values[i] - y_pre_.values[i])
    error /= len(y_pre)
    error /= y_pre.values.mean()

    ax.plot(y, c='g', label='train_set_actual_value')
    if args.method == 'scatter':
        ax.scatter(y_pre_.index, y_pre_, s=2, c='r', label='predict_value')
        ax.scatter(y_pre.index, y_pre, s=2, c='b', label='actual_value')
    if args.method == 'plot':
        ax.plot(y_pre_, c='r', label='predict_value')
        ax.plot(y_pre, c='b', label='actual_value')
    ax.set_xlabel('time')
    ax.set_ylabel(args.target)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
    plt.title(f'time vs {args.target}............error: {error}')
    plt.legend(loc='upper right')
    # plt.show()
    import os
    import re
    plt.savefig(os.path.join('../plots',
                             re.search('\w+', re.search('\w+.csv$',
                                                        args.location).
                                       group(0)).group(0) + '.pdf'))
