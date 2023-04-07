import os
import wikipedia
os.environ["MKL_NUM_THREADS"] = "1"

os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["OMP_NUM_THREADS"] = "1"


import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, render_template, jsonify
import yfinance as yf
from datetime import datetime
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.arima_model import ARIMA
import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
import sys
import io
import json
import chart_studio
import plotly
import plotly.graph_objects as go
import plotly.express as px
chart_studio.tools.set_credentials_file(username='patrickchirdon', api_key='erqaxYsjI9ZL6UF9SGe8')
import pandas as pd
import requests
from termcolor import colored as cl
import matplotlib.pyplot as plt
import os
import sys
import io
old_stdout = sys.stdout
new_stdout = io.StringIO()
sys.stdout = new_stdout
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm


plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression

# instantiate the Flask app.
app = Flask(__name__)

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route("/callback", methods=["POST","GET"])
def cb():
	return gm(request.args.get("data"))



# API Route for pulling the stock quote
@app.route("/quote")
def display_quote():

	#get the query string parameters
	symbol = request.args.get('symbol', default="AAPL")
	period = request.args.get('period', default="1y")
	interval = request.args.get('interval', default="1mo")

	#pull the quote
	quote = yf.Ticker(symbol)
	#use the quote to pull the historical data from Yahoo finance
	hist = quote.history(period=period, interval=interval)
	#convert the historical data to JSON
	data = hist.to_json()
	#return the JSON in the HTTP response
	return data





# Return the JSON data for the Plotly graph
def gm():
    df=pd.read_csv('myopenclose.csv')
    avg20=df.close.rolling(window=20, min_periods=1).mean()
    avg50=df.close.rolling(window=50, min_periods=1).mean()
    avg200=df.close.rolling(window=200, min_periods=1).mean()

    df5=pd.read_csv('new2.csv')
    mypredictions=df5['0']
    close2=df['close']
    close2.append(mypredictions)




    # plot the candlesticks
    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close']),go.Scatter(x=df['date'], y=avg200, name="avg200", line=dict(color='orange', width=1)),  go.Scatter(x=df['date'], y=avg20, name="avg20", line=dict(color='orange', width=1)), go.Scatter(x=df['date'], y=avg200, name="avg200", line=dict(color='orange', width=1)),
                      go.Scatter(x=df['date'], y=avg50, name="Avg50", line=dict(color='green', width=1))])



    fig.update_yaxes(
        title_text = "Price",
        title_standoff = 5)

    fig.update_xaxes(
        title_text = "Date",
        title_standoff = 5)

    fig.update_layout(title="Stock Price")



    # Create a JSON representation of the graph
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/')
def index():

   return render_template('homepage.html', graphJSON=gm())

@app.route('/insertintotable',methods = ['POST'])
def insertintotable():
    nm = request.form['nm']





#**************** FUNCTIONS TO FETCH DATA ***************************
    def get_historical(quote):
        end = datetime.now()
        start = datetime(end.year-2,end.month,end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)

        if(df.empty):
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            #Format df
            #Last 2 yrs rows => 502, in ascending order => ::-1
            data=data.head(503).iloc[::-1]
            data=data.reset_index()
            #Keep Required cols only
            df=pd.DataFrame()
            df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']

        return

    def get_historic_data(symbol):
        ticker = symbol
        iex_api_key = 'Tsk_30a2677082d54c7b8697675d84baf94b'
        api_url = f'https://sandbox.iexapis.com/stable/stock/{ticker}/chart/max?token={iex_api_key}'
        df = requests.get(api_url).json()

        date = []
        open = []
        high = []
        low = []
        close = []

        for i in range(len(df)):
            date.append(df[i]['date'])
            open.append(df[i]['open'])
            high.append(df[i]['high'])
            low.append(df[i]['low'])
            close.append(df[i]['close'])

        date_df = pd.DataFrame(date).rename(columns = {0:'date'})
        open_df = pd.DataFrame(open).rename(columns = {0:'open'})
        high_df = pd.DataFrame(high).rename(columns = {0:'high'})
        low_df = pd.DataFrame(low).rename(columns = {0:'low'})
        close_df = pd.DataFrame(close).rename(columns = {0:'close'})

        frames = [date_df, open_df, high_df, low_df, close_df]
        df = pd.concat(frames, axis = 1, join = 'inner')
        df = df.set_index('date')

        df.to_csv('myopenclose.csv')

        return df



    import matplotlib.pyplot as plt


  #***************** LINEAR REGRESSION SECTION ******************
    def LIN_REG_ALGO(df):
        #No of days to be forcasted in future
        forecast_out = int(7)
        #Price after n days
        df['Close after n days'] = df['close'].shift(-forecast_out)
        #New df with only relevant data
        df_new=df[['close','Close after n days']]

        #Structure data for train, test & forecast
        #lables of known data, discard last 35 rows
        y =np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
        #all cols of known data except lables, discard last 35 rows
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
        #Unknown, X to be forecasted
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])

        #Traning, testing to plot graphs, check accuracy
        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]

        # Feature Scaling===Normalization
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        X_to_be_forecasted=sc.transform(X_to_be_forecasted)

        #Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)

        #Testing
        y_test_pred=clf.predict(X_test)
        y_test_pred=y_test_pred*(1.04)
        import matplotlib.pyplot as plt2
        fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
        plt2.plot(y_test,label='Actual Price' )
        plt2.plot(y_test_pred,label='Predicted Price')

        plt2.legend(loc=4)
        plt2.savefig('static/LR.png')
        plt2.close(fig)

        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))


        #Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)

        forecast_set=forecast_set*(1.04)

        mean=forecast_set.mean()
        lr_pred=forecast_set[0,0]
        newlist=[]
        for i in forecast_set:
            i=np.round(i, decimals=2)
            newlist.append(i)
        forecast_set=newlist

        print()
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by Linear Regression: ",lr_pred)
        print("Linear Regression RMSE:",error_lr)
        print("##############################################################################")
        return df, lr_pred, forecast_set, mean, error_lr

        #Building RNN







    #**************GET DATA ***************************************
    quote=nm


    def get_simulation(ticker, name):

        data=pd.DataFrame()

        get_historic_data(ticker)
        myresults=pd.read_csv('myopenclose.csv')
        data[ticker]=myresults['close']

        log_returns=np.log(1 + data.pct_change())
        u=log_returns.mean()
        var=log_returns.var()
        drift=u-(0.5*var)
        stdev=log_returns.std()
        t_intervals=30
        iterations=10
        daily_returns=np.exp(drift.values  + stdev.values * norm.ppf(np.random.rand(t_intervals, iterations)))

        S0=data.iloc[-1]
        price_list=np.zeros_like(daily_returns)
        price_list[0]=S0
        for t in range(1, t_intervals):
            price_list[t]=price_list[t-1]* daily_returns[t]

        import matplotlib.pyplot as plty
        plty.figure(figsize=(10,6))
        plty.title('30 Day Monte Carlo Simulation')
        plty.ylabel('Price (P)')
        plty.xlabel('Time (Days)')
        plty.plot(price_list)
        plty.savefig('static/myfig.jpg')
        plty.clf()









    get_simulation(quote, ' + quote + ')

    #Try-except to check if valid stock symbol
    try:
        get_historic_data(quote)
        get_historical(quote)
    except:
        return render_template('homepage.html',not_found=True)
    else:

        #************** PREPROCESSUNG ***********************

        #df = pd.read_csv(''+quote+'.csv')


        df=pd.read_csv('myopenclose.csv')
        print("##############################################################################")
        print("Today's",quote,"Stock Data: ")
        today_stock=df.iloc[-1:]
        print(today_stock)
        print("##############################################################################")
        df = df.dropna()
        code_list=[]
        for i in range(0,len(df)):
            code_list.append(quote)
        df2=pd.DataFrame(code_list,columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df=df2



        df, lr_pred, forecast_set,mean,error_lr=LIN_REG_ALGO(df)
        print()
        print("Forecasted Prices for Next 7 days:")
        newa=df['date']
        newdf=df['close']
        newdf2a=df['open']
        newdf2ab=df['high']
        newdf2abc=df['low']
        #new3.csv
        myresults=pd.read_csv('myopenclose.csv')
        myvals=myresults['close']
        mylen=len(myvals)
        first=mylen-7
        thirtyseven=mylen-367
        myval2=myvals.to_numpy()

        firstset=myvals[thirtyseven:first]
        rollingaverage1=firstset.rolling(100, min_periods=1).mean()
        rollingaverage2=firstset.rolling(200, min_periods=1).mean()


        result = pd.concat([newa, newdf, newdf2a, newdf2ab, newdf2abc, rollingaverage1, rollingaverage2], axis=1, join='inner')
        result.columns=['timestamp', 'close','open','high','low','moving1','moving2']




        file = 'ScatterPlot_05.png'
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)


        file = 'ScatterPlot_0530day.png'
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)

        file = 'ScatterPlot_06.png'
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)

        file = 'ScatterPlot_0630day.png'
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)

        file = 'Holt.png'
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)

        file = 'Holt30day.png'
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)





        file = 'new2.csv'
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)

        file='static/ScatterPlot_05.png'
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)
        file='static/ScatterPlot_06.png'
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)


        newdf2=pd.DataFrame(forecast_set)






        newdf2.to_csv('new2.csv')
        frames=[newdf, newdf2]

        result=pd.concat(frames)



        myresults=pd.read_csv('myopenclose.csv')
        myvals=myresults['close']

        mylen=len(myvals)
        first=mylen-7
        thirtyseven=mylen-367
        myval2=myvals.to_numpy()

        firstset=myvals[thirtyseven:first]
        secondset=myvals[first:mylen]
        X=pd.Series(range(1,361))
        X2=pd.Series(range(361,368))
        #now calculate the 95% confidence interval
        mean=(myval2[0] + myval2[1] + myval2[2] + myval2[3] + myval2[4] + myval2[5] + myval2[6] + myval2[7] + myval2[8] + myval2[9] + myval2[10] + myval2[11] + myval2[12] + myval2[13] + myval2[14] + myval2[15] + myval2[16] + myval2[17] + myval2[18] + myval2[19] + myval2[20] + myval2[21] + myval2[22] + myval2[23] + myval2[24] + myval2[25] + myval2[26] + myval2[27] + myval2[28] + myval2[29] + myval2[30])/30
        sq1=(myval2[0]-mean)**2
        sq2=(myval2[1]-mean)**2
        sq3=(myval2[2]-mean)**2
        sq4=(myval2[3]-mean)**2
        sq5=(myval2[4]-mean)**2
        sq6=(myval2[5]-mean)**2
        sq7=(myval2[6]-mean)**2
        sq8=(myval2[7]-mean)**2
        sq9=(myval2[8]-mean)**2
        sq10=(myval2[9]-mean)**2
        sq11=(myval2[10]-mean)**2
        sq12=(myval2[11]-mean)**2
        sq13=(myval2[12]-mean)**2
        sq14=(myval2[13]-mean)**2
        sq15=(myval2[14]-mean)**2
        sq16=(myval2[15]-mean)**2
        sq17=(myval2[16]-mean)**2
        sq18=(myval2[17]-mean)**2
        sq19=(myval2[18]-mean)**2
        sq20=(myval2[19]-mean)**2
        sq21=(myval2[20]-mean)**2
        sq22=(myval2[21]-mean)**2
        sq23=(myval2[22]-mean)**2
        sq24=(myval2[23]-mean)**2
        sq25=(myval2[24]-mean)**2
        sq26=(myval2[25]-mean)**2
        sq27=(myval2[26]-mean)**2
        sq28=(myval2[27]-mean)**2
        sq29=(myval2[28]-mean)**2
        sq30=(myval2[29]-mean)**2
        inside=(sq1 + sq2 + sq3 + sq4 + sq5 + sq6 + sq7 + sq8 + sq9 + sq10 + sq11 + sq12 + sq13 + sq14 + sq15 + sq16 + sq17 + sq18 + sq19 + sq20 + sq21 + sq22 + sq23 + sq24 + sq25 + sq26 + sq27 + sq28 + sq29 + sq30)/30
        stdDev=math.sqrt(inside)

        import seaborn as snsg
        import matplotlib.pyplot as pltb

        myresults=pd.read_csv('myopenclose.csv')
        myvals=myresults['close']
        mylen=len(myvals)
        first=mylen-7
        thirtyseven=mylen-367
        myval2=myvals.to_numpy()


        firstset=myvals[thirtyseven:first]
        secondset=myvals[first:mylen]
        X=pd.Series(range(1,361))
        X2=pd.Series(range(361,368))
       # x=[361,362,363]
        #newarima1=arima_pred + error_arima
       # newarima2=arima_pred - error_arima
        #y=[arima_pred, newarima1, newarima2]
        #newy=pd.Series(y)
        #newx=pd.Series(x)

        snsg.regplot(X, firstset, ci=68,  color='red')
       # sns.regplot(newx, newy, ci=68)
        pltb.xlim(0,400)
        rollingaverage=firstset.rolling(100, min_periods=1).mean()

        snsg.regplot(X, rollingaverage, ci=68,  color='blue')

        snsg.regplot(X2, secondset,  color='purple')
        pltb.legend(labels =['Stock Price Scatter + Trendline', '100 Day Rolling Average + Trendline', 'Predicted'], fontsize=14)

        # title and labels
        pltb.xlabel('Day', fontsize=16)
        pltb.ylabel('Price', fontsize=16)
        pltb.title('Linear Regression 1 Year')
        pltb.savefig('static/ScatterPlot_05.png')


        pltb.clf()
        pltb.close()


        import matplotlib.pyplot as pltA
        import seaborn as snsq

        myresults=pd.read_csv('myopenclose.csv')
        myvals=myresults['close']
        first=mylen-7
        thirtyseven=mylen-37
        firstset2=myvals[thirtyseven:first]
        secondset2=myvals[first:mylen]
        X2=pd.Series(range(1,31))
        X22=pd.Series(range(31,38))
        #now calculate the 95% confidence interval
        mean=(myval2[0] + myval2[1] + myval2[2] + myval2[3] + myval2[4] + myval2[5] + myval2[6] + myval2[7] + myval2[8] + myval2[9] + myval2[10] + myval2[11] + myval2[12] + myval2[13] + myval2[14] + myval2[15] + myval2[16] + myval2[17] + myval2[18] + myval2[19] + myval2[20] + myval2[21] + myval2[22] + myval2[23] + myval2[24] + myval2[25] + myval2[26] + myval2[27] + myval2[28] + myval2[29] + myval2[30])/30
        sq1=(myval2[0]-mean)**2
        sq2=(myval2[1]-mean)**2
        sq3=(myval2[2]-mean)**2
        sq4=(myval2[3]-mean)**2
        sq5=(myval2[4]-mean)**2
        sq6=(myval2[5]-mean)**2
        sq7=(myval2[6]-mean)**2
        sq8=(myval2[7]-mean)**2
        sq9=(myval2[8]-mean)**2
        sq10=(myval2[9]-mean)**2
        sq11=(myval2[10]-mean)**2
        sq12=(myval2[11]-mean)**2
        sq13=(myval2[12]-mean)**2
        sq14=(myval2[13]-mean)**2
        sq15=(myval2[14]-mean)**2
        sq16=(myval2[15]-mean)**2
        sq17=(myval2[16]-mean)**2
        sq18=(myval2[17]-mean)**2
        sq19=(myval2[18]-mean)**2
        sq20=(myval2[19]-mean)**2
        sq21=(myval2[20]-mean)**2
        sq22=(myval2[21]-mean)**2
        sq23=(myval2[22]-mean)**2
        sq24=(myval2[23]-mean)**2
        sq25=(myval2[24]-mean)**2
        sq26=(myval2[25]-mean)**2
        sq27=(myval2[26]-mean)**2
        sq28=(myval2[27]-mean)**2
        sq29=(myval2[28]-mean)**2
        sq30=(myval2[29]-mean)**2
        inside=(sq1 + sq2 + sq3 + sq4 + sq5 + sq6 + sq7 + sq8 + sq9 + sq10 + sq11 + sq12 + sq13 + sq14 + sq15 + sq16 + sq17 + sq18 + sq19 + sq20 + sq21 + sq22 + sq23 + sq24 + sq25 + sq26 + sq27 + sq28 + sq29 + sq30)/30
        stdDev=math.sqrt(inside)
        confidence95= 1.960 * (stdDev / 5.47)
        confidence75=1.15*(stdDev/5.47)

        secondset2a=[float(str(np.round(j,2)).strip("[]")) for j in forecast_set]
        secondset3=[float(str(np.round((j- confidence95),2)).strip("[]")) for j in forecast_set]
        secondset4=[float(str(np.round((j+ confidence75),2)).strip("[]")) for j in forecast_set]
        secondset5=[float(str(np.round((j- confidence75),2)).strip("[]")) for j in forecast_set]



        snsq.regplot(X2, firstset2, ci=68, color='red')

        pltA.xlim(0,40)
        rollingaverage2=firstset2.rolling(100, min_periods=1).mean()

        snsq.regplot(X2, rollingaverage2, ci=68, color='blue')

        snsq.regplot(X22, secondset2, color='purple')
        pltA.legend(labels =['Stock Price + Trendline', '100 Day Rolling Average + Trendline', 'Predicted + 95% Confidence'], fontsize=14)


        # title and labels
        pltA.xlabel('Day', fontsize=16)
        pltA.ylabel('Price', fontsize=16)
        pltA.title('Linear Regression 30 Day')
        pltA.savefig('static/ScatterPlot_0530day.png')

        pltA.clf()

        pltA.close()












        return render_template('homepage.html', secondset5=secondset5, secondset2a=secondset2a, secondset3=secondset3, secondset4=secondset4,
                               open_s=today_stock['open'].to_string(index=False),
                               close_s=today_stock['close'].to_string(index=False),
                               high_s=today_stock['high'].to_string(index=False),
                               forecast_set=forecast_set,
                               low_s=today_stock['low'].to_string(index=False))



# run the flask app.
if __name__ == "__main__":
	app.run(debug=True)
