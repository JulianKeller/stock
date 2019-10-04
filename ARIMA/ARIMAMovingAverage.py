# Moving Average
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import datetime as dt


def set_date_as_index(data_frame):
    # converts the date column into datetime format and into the index
    data_frame['Date'] = pd.to_datetime(data_frame['Date'])
    data_frame.set_index('Date', inplace=True)
    return data_frame


def ARIMA_prediction(x_train):
    # determing the p and q values to use in ARIMA function
    # lag_acf = acf(x_train,nlags=10)
    # lag_pacf = pacf(x_train,nlags=10, method='ols')
    # plot ACF
    # plt.subplot(121)
    # plt.plot(lag_acf)
    # plt.axhline(y=0,linestyle='--',color='gray')
    # plt.axhline(y=-1.96/np.sqrt(len(x_train)),linestyle='--',color='gray')
    # plt.axhline(y=1.96/np.sqrt(len(x_train)),linestyle='--',color='gray')
    # plt.title('ACF')
    # plot PACF
    # plt.subplot(122)
    # plt.plot(lag_pacf)
    # plt.axhline(y=0,linestyle='--',color='gray')
    # plt.axhline(y=-1.96/np.sqrt(len(x_train)),linestyle='--',color='gray')
    # plt.axhline(y=1.96/np.sqrt(len(x_train)),linestyle='--',color='gray')
    # plt.title('PACF')
    # plt.tight_layout()
    # plt.show()

    # creating the ARIMA model
    model = ARIMA(x_train, order=(2, 1, 2))
    fit_ARIMA = model.fit(disp=-1)
    pred = fit_ARIMA.forecast()[0]
    # fitted_ARIMA = np.array(fit_ARIMA.fittedvalues, copy=True)
    # pred_dates = np.asarray(pd.date_range('9/26/2019',periods=22))
    # fitted = pd.DataFrame(values,index=pred_dates,columns = ['Pred Val'])
    # pred = model.predict(fit_ARIMA,start=[y_pred[0]],end=[y_pred[-1]])
    return pred


def plotting_single_file(x_train, plot_list):
    data = pd.read_csv(x_train)
    # set the index to be the dates
    data = set_date_as_index(data)
    # data_set = [[x] for x in data[plot_list[0]].values]
    # data_set = np.array(data_set)
    # data_length = len(data[plot_list[0]])
    dataset = data[plot_list[0]][-365:]
    x_train, y_pred = (train_test_split(dataset, test_size=0.1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = np.array(x_train)
    y_pred = np.array(y_pred)
    x_train = x_train.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    # x_train = scaler.fit_transform(x_train)
    # y_pred = scaler.fit_transform(y_pred)

    # pred = ARIMA_prediction(x_train,y_pred)
    # print (np.shape(y_pred))
    # print (np.shape(x_train))
    # pred = pred.reshape(-1,1)
    # train = scaler.inverse_transform(pred)
    actual = [x for x in x_train]
    predictions = list()
    for timepoint in range(len(x_train)):
        actual_val = x_train[timepoint]
        prediction = ARIMA_prediction(x_train)
        # print('Actual=%f, Predicted=%f' % (actual_val,prediction))
        predictions.append(prediction)
        actual.append(actual_val)

    # Error = mean_squared_error(x_train,prediction)
    # print('Test Mean Squared Error: %.3f' % Error)

    plt.plot(x_train)
    plt.plot(prediction, color='green')
    plt.show()


if __name__ == '__main__':
    plotting_single_file('../LSTM/data/costco/COST.csv', ['Close'])