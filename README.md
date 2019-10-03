# Machine Learning Project 1 - Stock Market Prediction
Team: JET=NaN


## TODO
Due Friday October 4
Summarized analysis with your results
    - Historical timeframe used by team
    - [ ] Explanation of logic for the specified timeframe
            - We picked 1 year of training data because it far less time consuming to traing the model. This also allowed us                to avoid large bumps/dips for stock seasonality.
    - [ ] Steps that you have taken to derive at result
        - All team members chose a machine learning model to implement. Then we compared the models prediction results to see           which one had the better outcome. We compared for the same Adidas stock for the same time period, and used the same           data for making the prediction.
        - Decided a timeframe for the stocks.
    - [ ] An explanation of the logic for each step
        - See Julians code comments, steps are similar to Blades results below: 
            - Considering target data as y and rest of data as x
            - Using MinMax Standardize the x set
            - Appling KNN model regression on it
            - Picking the best n_neighbors with 10 fold cross validation
            - Fit train data and get our prediction model
            - Predict data
    - [ ] Explanation of the responsibilities assigned to each team member
        - Each of us implemented a model. We all gathered stock data. 
- [ ] Download 3 more stocks data

Due November 6
- [ ] Make Comparison to actual stock prices, 
    
    
# Team Responsibilities
Each team member worked on a different machine learning algorithm.

## Erik - Moving Average


## Blade - K Nearest Neighbors


## Julian - Long Short Term Memory 


#### LSTM Sources Referenced
 - https://medium.com/neuronio/predicting-stock-prices-with-lstm-349f5a0974d4
 - https://heartbeat.fritz.ai/using-a-keras-long-shortterm-memory-lstm-model-to-predict-stock-prices-a08c9f69aa74
 - https://keras.io/getting-started/sequential-model-guide/
 - https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html
 - https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/
 - https://machinelearningmastery.com/make-predictions-long-short-term-memory-models-keras/
 - https://stackoverflow.com/questions/48760472/how-to-use-the-keras-model-to-forecast-for-future-dates-or-events
    
 #### LSTM Requirements:
 - Python3.6+
 - TensorFlow library
 - keras library
