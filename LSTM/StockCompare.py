import csv
import pandas as pd
import matplotlib.pyplot as plt
from os import path
import timeit

# convert csv's to a single pandas df
def read_files(actual, predicted, column):
    act = pd.read_csv(actual)
    pred = pd.read_csv(predicted)
    df = pd.DataFrame()
    length = len(pred['0'])
    df['Date'] = act['Date'].head(length)
    df['Actual'] = act[column].head(length)
    df['Predicted'] = pred['0']
    return df

# display the plots for the df
def display_graph(df, company, column):
    plt.plot(df['Date'], df['Actual'], label='Actual')
    plt.plot(df['Predicted'], label='Predicted')
    plt.title(f'{company} {column} Actual vs Predicted Stock')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($USD)')
    plt.legend()
    plt.xticks(rotation=55)
    # save the plot
    plot = plt.gcf()
    plt.show()
    plt.draw()
    return plot

# save the data
def save_predicted_csv(self, np_array):
    data = pd.DataFrame(np_array)
    print(data)
    data.to_csv(f"data/predictions/{self.name}-predicted.csv",)


if __name__ == '__main__':
    start = timeit.default_timer()
    columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    companies = [
        ['Adidas', 'ADDYY'],
        ['adp', 'ADP'],
        ['Bitcoin', 'BTC-USD'],
        ['Costco', 'COST'],
        ['FireEye', 'FEYE'],
        ['Gopro', 'GPRO'],
        ['Honeywell', 'HON'],
        ['medtronic', 'MDT'],
        ['s&p', 'S&P 500'],
        ['Tesla', 'TSLA'],
    ]

    for comp in companies:
        for col in columns:
            act = f'data/{comp[0].lower()}/{comp[1]}-30.csv'
            fut = f'data/predictions/{comp[0]}_{col}-predicted.csv'
            result = f'act_vs_pred/{comp[0]}_{col}_cmp.csv'
            data = read_files(act, fut, col)
            data.to_csv(result, sep=',')        # save the csv data

            plot_result = f'act_vs_pred/{comp[0]}_{col}_cmp.png'
            plot = display_graph(data, 'Adidas', col)
            plot.savefig(plot_result, dpi=100)

    print(f'Total run time: {timeit.default_timer() - start}')
