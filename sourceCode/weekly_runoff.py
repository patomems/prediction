import pandas as pd
import numpy as np
import math
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import time
import fbprophet
import matplotlib.pyplot as plt
import joblib


def weekly_runoff_forecast(filename, wtd):
    def import_data():
        raw_data_df = pd.read_excel('data/' + filename + '.xlsx', header=0)
        return raw_data_df

    raw_data_df = import_data()
    raw_data_df.head()
    raw_data_df['Date'] = pd.to_datetime(raw_data_df['Date'])
    for i in range(1, len(raw_data_df.columns)):
        raw_data_df[raw_data_df.columns[i]] = raw_data_df[raw_data_df.columns[i]].fillna(
            raw_data_df[raw_data_df.columns[i]].mean())
        data = pd.DataFrame()
        data['Date'] = raw_data_df["Date"]
        data['weekly runoff'] = raw_data_df["weekly runoff"]
        data = data.set_index(['Date'])
        data.head()
        data.isnull().sum()
        data.dropna().describe()
        monthly = data.resample('M').sum()
        monthly.plot(style=[':', '--', '-'], title='Monthly Trends')
        weekly = data.resample('W').sum()
        daily = data.resample('D').sum()
        daily.head()
        values = daily['weekly runoff'].values.reshape(-1, 1)
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        scale = daily
        scale["weekly runoff"] = scaled
        scale.head()
        scale.shape

        def making_dataset(i=1):
            if i == 0:
                df1 = scale.iloc[6940:, :]
                df2 = scale.iloc[:6940, :]
                df2.reset_index(inplace=True)
                df2 = df2.rename(columns={'Date': 'ds', 'weekly runoff': 'y'})
                return df2, df2
            else:
                df2 = scale.iloc[:, :]
                df2.reset_index(inplace=True)
                df2 = df2.rename(columns={'Date': 'ds', 'weekly runoff': 'y'})
                return df2, df2

            df1, df2 = making_dataset(wtd)
            df2.head()

            import warnings
            warnings.simplefilter(action='ignore', category=FutureWarning)
            path = 'trained/' + filename + '_weekly_runoff'
            df2_prophet = joblib.load(path + '.pkl')
            warnings.resetwarnings()

            def predicting_data(i=1):
                if i == 0:
                    df2_forecast = df2_prophet.make_future_dataframe(periods=30 * 25, freq='D')
                    df2_forecast = df2_prophet.predict(df2_forecast)
                    df3 = df2_forecast[['ds', 'yhat']]
                    df3.shape, df1.shape, df2.shape
                    df4 = df3.iloc[6940:-20, :]
                else:
                    df2_forecast = df2_prophet.make_future_dataframe(periods=30 * 12, freq='D', include_history=False)
                    df2_forecast = df2_prophet.predict(df2_forecast)
                    df3 = df2_forecast[['ds', 'yhat']]
                    df4 = df3.iloc[:, :]
                return df4, df2_forecast

            df4, df2_forecast = predicting_data(wtd)
            ypred = df4.iloc[:, 1:]
            ytest = df1.iloc[:, :]
            ypred.shape, ytest.shape

            df4.tail()
            ypred = df4.iloc[:, 1:]
            ytest = df1.iloc[:, :]
            ypred.shape, ytest.shape

            from sklearn.metrics import mean_absolute_error
            if wtd == 0:
                print("mean_absolute_error=", mean_absolute_error(ytest, ypred))
                df2_prophet.plot(df2_forecast, xlabel='Date', ylabel='weekly runoff')
                plt.title('simple test');
                df2_prophet.plot_components(df2_forecast)
                df4.columns = ['Date', 'weekly runoff']
                values = df4['weekly runoff'].values.reshape(-1, 1)
                values = values.astype('float32')
                valu = scaler.inverse_transform(values)
                df4['weekly runoff'] = valu
                df4['weekly runoff'] = abs(df4['weekly runoff'])
                df4.to_csv('data/forecast/' + filename + '_weekly_runoff_forecast.csv', index=False)
                return df4
