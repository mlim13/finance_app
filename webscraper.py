'''
- we are scraping a list of S&P500 company tickers 
- S&P500 is just a list of the most valuable companies
'''

import bs4 as bs # ensures if beautiful soup gets updated we dont have to change lots of code
import pickle
import requests
import pandas as pd
import pandas_datareader as web
import datetime as dt
import os
from collections import Counter
import numpy as np
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


def save_snp():
    # first we need to make a request to a webpage
    # we want to get the source code of the webpage for us to subsequently parse
    response = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies") # just a wiki article that lists all the companies
    soup = bs.BeautifulSoup(response.text, "lxml") # we create a bs object with the text of the source code
    table = soup.find("table", {"class":"wikitable sortable"})
    tickers = []

    for row in table.findAll("tr")[1:]: # table.findAll("tr") returns an iterable of all rows (tr is row tag). The first row is just headers so we skip that
        ticker = row.findAll("td")[0].text # of the row, we want the 0th data tag (first column)
        ticker = ticker.replace(".", "-") # some tickers have format a.b. Yahoo uses format a-b
        tickers.append(ticker[:-1]) # we want to append the text version not the soup object version. Also removing newline char

    # now we want to save this list to a file
    with open("snp_tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers


def get_data(reload = False): # reload is an argument that tells us if we want to reload the tickers or use the existing ticker file
    # we now want a means of getting the stock data for each of the tickers
    if reload:
        tickers = save_snp()
    else:
        with open("snp_tickers.pickle", "rb") as f:
            tickers = pickle.load(f) 

    #creating a folder to hold all our stock data
    if not os.path.exists("stock_data"):
        os.makedirs("stock_data")

    start = dt.datetime(2015, 1, 1)
    end = dt.datetime(2019, 1, 1)
    for ticker in tickers:
        print(ticker)
        if not os.path.exists("stock_data/{}".format(ticker)):
            try:
                df = web.DataReader(ticker, "yahoo", start, end)
                df.to_csv("stock_data/{}".format(ticker))
            except:
                print("No data for this atm")
        else:
            print("Already have {}.".format(ticker))

def combine_data():
    # we want to take the snp data and build a compiled dataframe for a single column (eg. adj close for all companies)
    main_df = pd.DataFrame() # creating an empty dataframe
    
    with open("snp_tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    for ticker in tickers:
        if not os.path.exists("stock_data/{}.csv".format(ticker)):
            continue
        df = pd.read_csv("stock_data/{}.csv".format(ticker))
        df.set_index("Date", inplace = True)
        df.rename(columns = {"Adj Close":ticker}, inplace = True)
        df.drop(["Open", "High", "Low", "Close", "Volume"], 1, inplace = True)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how = "outer")
    
    main_df.to_csv("adj_close_data.csv")

def preprocess_data_for_ml(ticker):
    # for a given company, we will process the data in order to subsequently train it
    # for a given date, we want to see if IN THE FUTURE the price rises or falls
    # thus, we add columns to our df showing the %rise/fall for eac day into the future
    # note, for a df, operations can be done for entire series (columns of data)
    df = pd.read_csv("adj_close_data.csv", index_col = "Date")
    tickers = df.columns.values.tolist() # we dont actually need this in this function but will use later so we return it
    df.fillna(0, inplace = True)

    num_days = 5 # how many days into the future fo we want to use
    for i in range(1, num_days + 1):
        df["{}_{}d".format(ticker, i)] = (df.shift(-i)[ticker] - df[ticker]) / df[ticker]
  
    df.fillna(0, inplace = True)

    return tickers, df

def buy_sell_hold(*args):
    # we will be passing a bunch of columns to this function
    # each column is one of the columns produced by preprocess_data_for_ml(ticker)
    requirement = 0.02 # this is the percentage increase that defines whether we buy sell or hold
    for col in args:
        if col > requirement:
            return 1 # buy
        elif col < -requirement:
            return -1 # sell
    return 0 # hold

def map_function(ticker):
    # here we will append a new column to our dataframe with the label of buy sell or hold
    tickers, df = preprocess_data_for_ml(ticker)
    df["{}_label".format(ticker)] = list(map(buy_sell_hold, df["{}_1d".format(ticker)], df["{}_2d".format(ticker)], df["{}_3d".format(ticker)], df["{}_4d".format(ticker)], df["{}_5d".format(ticker)]))

    # we want to see the spread of labels generated
    vals = df["{}_label".format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print("Data Spread:", Counter(str_vals)) # Counter is just a pre-built way of counting occurrences

    # lets remove any bad data
    df.fillna(0, inplace = True)
    df = df.replace([np.inf, np.NINF], np.nan)
    df.dropna(inplace = True)

    # now we want to define our features and our labels
    # remember, we are using the "future" data to make our labels but when we actually train our model, we need to train off the data ITSELF

    df_vals = df[[ticker for ticker in tickers]].pct_change() # creates percentage change from prev to curr
    df_vals = df.replace([np.inf, np.NINF], 0)
    df_vals.fillna(0, inplace =True)
    X = df_vals.values
    y = df["{}_label".format(ticker)].values

    return X, y, df

def do_ml(ticker):
    X, y, df = map_function(ticker)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25)
    print(X_train)
    #print(max(X_train))
    #print(min(X_train))
    print(y_train)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("Predicted Spread:", Counter(predictions))
    confidence = clf.score(X_test, y_test)
    print("Accuracy:", confidence)

    return confidence
     

if __name__ == "__main__":
    #save_snp() 
    #get_data()
    #combine_data()
    #buy_sell_hold("hello", "bye")
    #preprocess_data_for_ml("BAC")
    #map_function("BAC")
    do_ml("BAC")
    