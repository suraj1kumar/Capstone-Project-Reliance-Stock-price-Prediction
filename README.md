# Reliance Industries Stock Price Prediction
![StockMarker](https://github.com/user-attachments/assets/efc14acb-7c2f-42c4-aca0-a275dec147ea)
This is the capstone project submitted to the Asian School of Media Studies in partial fulfillment of the requirements for the award of the Diploma in Data Science by Suraj Kumar Pandit under the supervision of Prof. Manpreet Kaur Bhatia.

# Project Overview
This project aims to predict the stock prices of Reliance Industries Limited using machine learning techniques, specifically the Random Forest algorithm. The goal is to provide accurate stock price predictions to aid investors in making informed decisions.

# Table of Contents
Introduction
Business Objective
Data Collection
Data Preprocessing
Model Selection and Implementation
Results and Discussion
Conclusion
Future Scope of Work
References
Introduction
# Background and Motivation
The stock market is a complex, dynamic, and influential component of the global economy. Predicting stock prices has always been challenging due to market volatility and numerous influencing factors. Traditional methods often fall short in capturing intricate patterns. This project leverages machine learning, particularly the Random Forest algorithm, to improve prediction accuracy.

# Problem Statement
The project aims to use Random Forest to predict the closing prices of Reliance Industries Limited's stock. The primary challenge is handling the volatile nature of stock prices and ensuring the model generalizes well to new data.

# Objectives
Preprocess and analyze historical stock price data.
Engineer relevant features to enhance the model's predictive capability.
Build and optimize a Random Forest model for accurate stock price prediction.
Evaluate the model's performance using appropriate metrics.
# Business Objective
Predict the Reliance Industries stock price for the next 30 days.
Obtain Open, High, Low, and Close prices from the web for each day from 2015 to 2022.
Split the last year into a test set to build a model for stock price prediction.
Identify short-term and long-term trends.
Understand the impact of external factors and significant events.
Forecast stock prices for the next 30 days.
Data Collection
For this project, the Yfinance library was used to collect data from 1-Jan-2015 to 28-Feb-2023. Data can also be downloaded from Yahoo! Finance.
 
# About the Data
Date: Date of trade
Open: Opening price of stock
High: Highest price of stock on that day
Low: Lowest price of stock on that day
Close: Close price adjusted for splits
Adj Close: Adjusted close price adjusted for splits and dividend/capital gain distributions
Volume: Volume of stock on that day
# Data Preprocessing
Data preprocessing involved cleaning the dataset, handling missing values, and normalizing the data to prepare it for analysis.

# Model Selection and Implementation
# Model Selection
The Random Forest algorithm was chosen due to its robustness, flexibility, and ability to handle large datasets with high dimensionality.

# Implementation
The model was trained on historical stock data, and relevant features were engineered to improve predictive performance. The implementation steps include:

Training the model using historical data.
Testing the model on a separate test set.
Evaluating the model's performance using metrics such as RMSE and MAE.
Results and Discussion
The results of the model were analyzed to determine its accuracy and effectiveness. The model's predictions were compared with actual stock prices, and the performance was evaluated using various metrics.

# Conclusion
The project demonstrated the potential of machine learning, specifically Random Forest, in predicting stock prices. Accurate stock price prediction can lead to better investment strategies, risk management, and economic stability.

# Future Scope of Work
Future work could explore other machine learning algorithms, incorporate more features, and improve the model's performance. Further research can also investigate the impact of different external factors on stock prices.


