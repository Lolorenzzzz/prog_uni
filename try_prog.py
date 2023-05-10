# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:00:15 2023

@author: loren
"""

import pandas as pd
import matplotlib.pyplot as plt
import ccxt
import time
import statsmodels.api as sm
import yfinance as yf
import statsmodels.api as sm
import warnings
from statsmodels.graphics.tsaplots import plot_pacf
import datetime
from datetime import date, timedelta
import sys


#check user input, loop until its valid
while True:
    a = input("Enter 1 to analyze a specific cryptocurrency, or 0 to see all available options: ")
    if (a == "0"):
       
        # Create a ccxt exchange object
        exchange = ccxt.binance()
    
        # Load all available markets and symbols
        markets = exchange.load_markets()
        symbols = list(markets.keys())
    
        # Print the list of symbols
        print(symbols)
        break
        
    elif(a == "1"):
        break
    
    else:
        print("Your input isn't valide")
    
    
    



# Load cryptocurrency price data from Yahoo Finance
symbol = input("Enter the symbol of the cryptocurrency you want to analyze (in this format: BTC-USD): ")
        
    


        
#set the timeline for loading the data
today = date.today()
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=365)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2



#get the data from yahoo finance
df = yf.download(symbol, 
                      start=start_date, 
                      end=end_date, 
                      progress=False)


#symbol = 'BTC-USD'
#start_date = '2019-01-01'

#df = yf.download(symbol, start=start_date)
#df.dropna(inplace=True)

#prepare the data for the analyisis
df["Date"] = df.index
df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
df.reset_index(drop=True, inplace=True)
df = df[["Date", "Close"]]

#check if user wanna see latest closign stockprices
st_p = input("if you wanna see the latest closing stock prices hit: y ")
if st_p == "y":
    print(df.tail())

#check if user wanna see plot of the historical stockprices
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.plot(df["Date"], df["Close"])
#check if user wanna see plot of the historical stockprices
p_hist = input("if you wanna see the plot of historical stock prices hit: y ")
if p_hist == "y":
    plt.show()

#set default values p, q, d for the model (d can't be changed bc stockdata is seasonal)
p, q, d = 5, 2, 1

#plot autocorrelation: choose p
pd.plotting.autocorrelation_plot(df["Close"])
#check if user wanna adjust p value
pshow = input("if you wanna see the autocorrelation plot to assign a p value (otherwise default is used) hit: y ")

#if yes show the plot and loop until user enters a valid input
if (pshow == "y"):
    plt.show()
    while True:
        try:
            p = int(input("Choose your p value (use provided autocorrelation plot): "))
            break
        except ValueError:
            print("Input has to be an integer")




#plot partial autocorrelation: choose q
plot_pacf(df["Close"], lags = 100)
#check if user wanna adjust q value
p2show = input("if you wanna see the autocorrelation plot to assign a q value (otherwise default is used) hit: y ")

#if yes show the plot and loop until user enters a valid input
if (p2show == "y"):
    plt.show()
    while True:
        try:
            q = int(input("Choose your q value (use provided partial autocorrelation plot): "))
            break
        except ValueError:
            print("Input has to be an integer")




#performing the timeseries forecasting
#setting up the model
model=sm.tsa.statespace.SARIMAX(df['Close'],
                                order=(p, d, q),
                                seasonal_order=(p, d, q, 12))

#first train it
model=model.fit()
#print statistics
print(model.summary())

#then do the forecasting
predictions = model.predict(len(df), len(df)+10)
print(predictions)


#print the plot of the historical prices combined with the predicted
df["Close"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")






