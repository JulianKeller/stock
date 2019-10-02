from time import time

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
    def __init__(self, name, timestep, epoch, batch, output_dim, dropout, data_column, csv_train_file, csv_test_file,
                 csv_future):
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
        print(future.shape)
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
        X_data = []  # TODO refactor name

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
        # training_set = self.training_data.iloc[:, 4:5].values  # get the close values
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

    # make a prediction based on csv_test
    def predict(self):
        # test the model
        dataset_test = pd.read_csv(self.csv_test)  # import the test set that we will make predictions on
        self.dataset_test = self.set_date_as_index(dataset_test)
        real_stock_price = [[x] for x in dataset_test[self.data_column].values]
        real_stock_price = np.array(real_stock_price)

        # merge the training and test sets on axis -
        dataset_total = pd.concat((self.training_data[self.data_column], dataset_test[self.data_column]), axis=0)

        # transform the new dataset for performance
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - self.timestep:].values
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
        test_predict_price = self.scaler.inverse_transform(test_predict_price)    # rescale prices

        # predict the future
        future = self.predict_future()
        future = self.offset_array(future, len(real_stock_price))

        # plot the data
        plt.plot(real_stock_price, color='grey', label=f'{self.name} Stock Price')
        plt.plot(test_predict_price, label=f'Predicted {self.name} Stock Price')
        plt.plot(future, label=f'Predicted {self.name} Future Stock Price')
        plt.title(f'{self.name} Test Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(f'{self.name} Stock Price')
        plt.legend()

        # save the plot
        timestamp = int(time())  # time since epoch
        plot = plt.gcf()
        plt.show()
        plt.draw()
        plot.savefig(f'plot_{self.name}_{self.epoch}_{timestamp}.png', dpi=100)

    def run_lstm(self):
        self.get_training_data()
        self.build_model()
        self.predict()


if __name__ == '__main__':
    lstm = LongShortTermMemory(name='costco',
                               timestep=7,
                               epoch=20,
                               batch=7,
                               output_dim=50,
                               dropout=20,
                               data_column='Close',
                               csv_train_file='data/costco.csv',
                               csv_test_file='data/costco-4weeks.csv',
                               csv_future='data/costco-future.csv')
    lstm.run_lstm()
