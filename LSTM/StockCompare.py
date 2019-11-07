import csv
import pandas as pd
import matplotlib.pyplot as plt

# TODO
# read in the predicted vs the actual
# Take two columns of data Open,High,Low,Close,Adj Close
# create plots comparing predicted/actual
# output columns with before and after


def compare_columns(col1, col2):
    pass

# TODO output future calculated data from program
# line 176: future = self.predict_future()

def read_files(actual, predicted):
    columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    act = pd.read_csv(actual)
    pred = pd.read_csv(predicted)
    df = pd.DataFrame()
    df['Date'] = act['Date']
    df['Actual'] = act['Close']
    df['Predicted'] = pred['Close']
    return df

def display_graph(df, company):
    plt.plot(df['Date'], df['Actual'], label='Actual')
    plt.plot(df['Predicted'], label='Predicted')
    plt.title(f'{company} Actual vs Predicted Stock')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($USD)')
    plt.legend()
    plt.xticks(rotation=55)
    # save the plot
    plot = plt.gcf()
    plt.show()
    plt.draw()


if __name__ == '__main__':
    act = 'data/adidas/ADDYY-30.csv'
    fut = 'data/adidas/ADDYY-future.csv'
    data = read_files(act, fut)
    display_graph(data, 'Adidas')