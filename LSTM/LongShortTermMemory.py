from time import time
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


# Sources Referenced
# https://medium.com/neuronio/predicting-stock-prices-with-lstm-349f5a0974d4
# https://heartbeat.fritz.ai/using-a-keras-long-shortterm-memory-lstm-model-to-predict-stock-prices-a08c9f69aa74
# https://keras.io/getting-started/sequential-model-guide/
# https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html
# https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/
# https://machinelearningmastery.com/make-predictions-long-short-term-memory-models-keras/
# https://stackoverflow.com/questions/48760472/how-to-use-the-keras-model-to-forecast-for-future-dates-or-events

# Requirements:
# Python3.6+
# TensorFlow library
# keras library


class LongShortTermMemory:
    def __init__(self, name, timestep, epoch, batch, output_dim, dropout, data_column, csv_train_file, csv_future,
                 csv_test_file=None, csv_actual_future=None):
        self.name = name
        self.timestep = timestep
        self.epoch = epoch
        self.batch = batch
        self.output_dim = output_dim
        self.dropout_percent = dropout / 100
        self.data_column = data_column
        self.csv_train = csv_train_file
        self.csv_test = csv_test_file
        self.csv_future = csv_future
        self.csv_actual_future = csv_actual_future
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = None
        self.y_train = None
        self.model = None
        self.training_data = None

    # array is an np array
    def max_range(self, array):
        # determine max size of the data based as a multiple of the batch size
        max_range = int((len(array) - self.timestep) / self.batch)
        max_range = max_range * self.batch + self.timestep
        return max_range

    # reshape the array
    def reshape_2d_array(self, array):
        max_range = self.max_range(array)
        array_3d = []
        for i in range(self.timestep, max_range):
            array_3d.append(array[i - self.timestep:i, 0])
        array_3d = np.array(array_3d)
        array_3d = np.reshape(array_3d, (array_3d.shape[0], array_3d.shape[1], 1))
        return array_3d

    # offset a numpy array for displaying the graph correctly
    def offset_array(self, array, offset):
        offset_arr = []
        for i in range(offset):
            offset_arr.append([None])
        for i in range(len(array)):
            offset_arr.append(array[i])
        offset_arr = np.array(offset_arr)
        return offset_arr

    # predicts the future!
    def predict_future(self):
        # predict based on the last few weeks
        X_data = self.get_data(self.csv_future)
        predictions = self.model.predict(X_data)
        future = [predictions]
        future = np.array(future)
        future = self.scaler.inverse_transform(future[0])  # rescale prices
        return future

    # use the pandas dataframe as the index
    @staticmethod
    def set_date_as_index(data_frame):
        data_frame['Date'] = pd.to_datetime(data_frame['Date'])
        data_frame.set_index('Date', inplace=True)
        return data_frame

    # get data from a csv file
    def get_data(self, csv):
        # import the close data
        data = pd.read_csv(csv)
        # set the index to be the dates
        data = self.set_date_as_index(data)
        data_set = [[x] for x in data[self.data_column].values]
        data_set = np.array(data_set)
        # scale the data for performance
        data_set_scaled = self.scaler.fit_transform(data_set)
        # put data into 3d array for LSTM digestion
        X_data = []
        # determine max size of the data based as a multiple of the batch size
        max_range = self.max_range(data)
        for i in range(self.timestep, max_range):
            X_data.append(data_set_scaled[i - self.timestep:i, 0])
        # convert the data to numpy arrays
        X_data = np.array(X_data)
        # reshape the data for the LSTM model
        X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))
        return X_data

    def get_training_data(self):
        # import the close data
        self.training_data = pd.read_csv(self.csv_train)
        self.training_data = self.set_date_as_index(self.training_data)
        # get the close values
        training_set = [[x] for x in self.training_data[self.data_column].values]
        training_set = np.array(training_set)
        # scale the data for performance
        training_set_scaled = self.scaler.fit_transform(training_set)
        # put data into 3d array for LSTM digestion
        X_train = []
        y_train = []
        for i in range(self.timestep, len(self.training_data)):
            X_train.append(training_set_scaled[i - self.timestep:i, 0])
            y_train.append(training_set_scaled[i, 0])
        # convert the data to numpy arrays
        X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_train = np.reshape(X_train,
                                  (X_train.shape[0], X_train.shape[1], 1))  # reshape the data for the LSTM model

    # Build the LSTM model
    def build_model(self):
        model = Sequential()  # initialize the model
        # add layers to our model
        model.add(LSTM(units=self.output_dim, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        model.add(Dropout(self.dropout_percent))
        model.add(LSTM(units=self.output_dim, return_sequences=True))
        model.add(Dropout(self.dropout_percent))
        model.add(LSTM(units=self.output_dim, return_sequences=True))
        model.add(Dropout(self.dropout_percent))
        model.add(LSTM(units=self.output_dim))
        model.add(Dropout(self.dropout_percent))
        model.add(Dense(units=1))  # add layer to specify output of 1 layer
        # compile with the Adam optimizer and computer the mean squared error
        model.compile(optimizer='adam', loss='mean_squared_error')
        # run the model, this may take several minutes
        model.fit(self.X_train, self.y_train, epochs=self.epoch, batch_size=self.batch)
        self.model = model

    # save the predicted data as a csv file
    def save_predicted_csv(self, np_array):
        data = pd.DataFrame(np_array)
        print(data)
        data.to_csv(f"data/predictions/{self.name}-predicted.csv",)

    # make a prediction based on csv_test
    def predict(self):
        # test the model
        dataset_test = pd.read_csv(self.csv_test)  # import the test set that we will make predictions on
        self.dataset_test = self.set_date_as_index(dataset_test)
        real_stock_price = [[x] for x in dataset_test[self.data_column].values]
        real_stock_price = np.array(real_stock_price)
        dataset_total = self.training_data[self.data_column]
        # transform the new dataset for performance
        inputs = dataset_total.values
        inputs = inputs.reshape(-1, 1)
        inputs = self.scaler.transform(inputs)
        # Reshape the data to a 3d array
        X_test = []
        for i in range(self.timestep, len(dataset_test)):
            X_test.append(inputs[i - self.timestep:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        # make the prediction
        test_predict_price = self.model.predict(X_test)
        test_predict_price = self.scaler.inverse_transform(test_predict_price)  # rescale prices
        # predict the future
        future = self.predict_future()
        # save predicted data to a csv
        self.save_predicted_csv(future)

        # future = self.offset_array(future, len(real_stock_price))


        # TODO uncomment this to output and save the plots
        # actual future data
        # if self.csv_actual_future is not None:
        #     actual_future = pd.read_csv(self.csv_actual_future)  # import the test set that we will make predictions on
        #     self.actual_future = self.set_date_as_index(actual_future)
        #     actual_future_stock_price = [[x] for x in actual_future[self.data_column].values]
        #     actual_future_stock_price = np.array(actual_future_stock_price)
        #     actual_future_stock_price = self.offset_array(actual_future_stock_price, len(real_stock_price))
        #
        # # # plot the data
        # plt.plot(real_stock_price, color='darkgrey', label=f'{self.name} Stock Price')
        # plt.plot(test_predict_price, color='orange', label=f'Predicted {self.name} Stock Price')
        # plt.plot(future, color='darkviolet', label=f'Predicted {self.name} Future Stock Price')
        # if self.csv_actual_future is not None:
        #     plt.plot(actual_future_stock_price, color='green', label=f'{self.name} Actual Future Stock Price')
        # plt.title(f'{self.name} Test Price Prediction')
        # plt.xlabel('Time')
        # plt.ylabel(f'{self.name} Stock Price')
        # plt.legend()
        # # save the plot
        # timestamp = int(time())  # time since epoch
        # plot = plt.gcf()
        # plt.show()
        # plt.draw()
        # if self.csv_actual_future is not None:
        #     plot.savefig(f'future_vs_actual_plots/af_{self.name}_{self.epoch}_{timestamp}.png', dpi=100)
        # else:
        #     plot.savefig(f'plots/{self.name}_{self.epoch}_{timestamp}.png', dpi=100)

    def run_lstm(self):
        self.get_training_data()
        self.build_model()
        self.predict()


if __name__ == '__main__':
    # uncomment one section below to run the model for those companies
    # list of stock files
    companies = ['Adidas', 'Bitcoin', 'Costco', 'S&P 500', 'ADP', 'Honeywell', 'Medtronic', 'FireEye', 'GoPro', 'Tesla']
    train_stocks = ['data/adidas/ADDYY.csv', 'data/bitcoin/BTC-USD.csv', 'data/costco/COST.csv', 'data/s&p/^GSPC.csv',
                    'data/adp/ADP.csv', 'data/honeywell/HON.csv', 'data/medtronic/MDT.csv',
                    'data/fireeye/FEYE.csv', 'data/gopro/GPRO.csv', 'data/tesla/TSLA.csv']
    test_stocks = ['data/adidas/ADDYY.csv', 'data/bitcoin/BTC-USD.csv', 'data/costco/COST.csv', 'data/s&p/^GSPC.csv',
                   'data/adp/ADP.csv', 'data/honeywell/HON.csv', 'data/medtronic/MDT.csv',
                   'data/fireeye/FEYE.csv', 'data/gopro/GPRO.csv', 'data/tesla/TSLA.csv']
    future_stocks = ['data/adidas/ADDYY-future.csv', 'data/bitcoin/BTC-USD-future.csv', 'data/costco/COST-future.csv',
                     'data/s&p/^GSPC-future.csv',
                     'data/adp/ADP-future.csv', 'data/honeywell/HON-future.csv', 'data/medtronic/MDT-future.csv',
                                                                                 'data/fireeye/FEYE-future.csv',
                     'data/gopro/GPRO-future.csv', 'data/tesla/TSLA-future.csv']
    actual_future_stocks = ['data/adidas/ADDYY-30.csv', 'data/bitcoin/BTC-USD-test.csv', 'data/costco/COST-test.csv',
                            'data/s&p/^GSPC-test.csv',
                            'data/adp/ADP-30.csv', 'data/honeywell/HON-30.csv', 'data/medtronic/MDT-30.csv',
                            'data/fireeye/FEYE-30.csv', 'data/gopro/GPRO-30.csv', 'data/tesla/TSLA-30.csv']



    # companies = ['Medtronic', 'FireEye', 'GoPro', 'Tesla']
    # train_stocks = ['data/medtronic/MDT.csv', 'data/fireeye/FEYE.csv', 'data/gopro/GPRO.csv', 'data/tesla/TSLA.csv']
    # test_stocks = ['data/medtronic/MDT.csv', 'data/fireeye/FEYE.csv', 'data/gopro/GPRO.csv', 'data/tesla/TSLA.csv']
    # future_stocks = ['data/medtronic/MDT-future.csv', 'data/fireeye/FEYE-future.csv', 'data/gopro/GPRO-future.csv', 'data/tesla/TSLA-future.csv']
    # actual_future_stocks = ['data/medtronic/MDT-30.csv','data/fireeye/FEYE-30.csv', 'data/gopro/GPRO-30.csv', 'data/tesla/TSLA-30.csv']

    columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    start = timeit.default_timer()
    for i in range(len(train_stocks)):
        for col in columns:
            lstm = LongShortTermMemory(name=f'{companies[i]}_{col}',
                                       timestep=7,
                                       epoch=100,
                                       batch=7,
                                       output_dim=50,
                                       dropout=20,
                                       data_column=col,
                                       csv_train_file=train_stocks[i],
                                       csv_test_file=test_stocks[i],
                                       csv_future=future_stocks[i],
                                       csv_actual_future=actual_future_stocks[i])
            lstm.run_lstm()
    print(f'Total run time: {timeit.default_timer() - start}')
