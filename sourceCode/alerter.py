import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import plotly.graph_objs as go
import chart_studio.plotly as py
import plotly
import warnings

warnings.filterwarnings("ignore")


def alerting():
    data1 = pd.read_csv(
        "/home/boyking/project /Rainfall_Flood/sourceCode/data/forecast"
        "/forecasted_level_of_rivers.csv")
    res = []
    for i in range(data1.shape[1]):
        for j in range(data1.shape[0]):
            if data1.iloc[j, i] == 1:
                res.append(data1.columns[i])
                break
    return res


def water_level_predictior():
    filename = ['Cauvery', 'Godavari', 'Krishna', 'Mahanadi', 'Son']

    def flood_classifier(filename, data, validating=0):

        data1 = pd.read_excel('data/' + filename + '.xlsx')

        data1.shape

        for i in range(1, len(data1.columns)):
            data1[data1.columns[i]] = data1[data1.columns[i]].fillna(data1[data1.columns[i]].mean())
        data1.describe()
        y = data1['Flood']
        for i in range(len(y)):
            if (y[i] >= 0.1):
                y[i] = 1

        y = pd.DataFrame(y)

        data1.drop('Flood', axis=1, inplace=True)

        data1.drop('Date', inplace=True, axis=1)

        x_train = data1
        y_train = y
        x_test = data

        sm = SMOTE(random_state=2)
        X_train_res, Y_train_res = sm.fit_sample(x_train, y_train)
        X_train_res.shape
        x_train, y_train = shuffle(X_train_res, Y_train_res, random_state=0)

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        clf1 = LinearDiscriminantAnalysis()
        clf1.fit(x_train, y_train)
        y_predict = clf1.predict(x_test)
        print(set(y_predict))
        return y_predict

    def dataCreator(filename):
        filename = None
        data1 = pd.read_csv('data/forecast/' + filename + '_discharge_forecast.csv')
        data2 = pd.read_csv('data/forecast/' + filename + '_flood_runoff_forecast.csv')
        data3 = pd.read_csv('data/forecast/' + filename + '_daily_runoff_forecast.csv')
        data4 = pd.read_csv('data/forecast/' + filename + '_weekly_runoff_forecast.csv')

        data1.shape, data2.shape, data3.shape, data4.shape

        data3.head()

        data = data1
        data['flood runoff'] = data2['flood runoff']
        data['daily runoff'] = data3['daily runoff']
        data['weekly runoff'] = data4['weekly runoff']
        data.head()

        for i in range(1, len(data.columns)):
            data[data.columns[i]] = data[data.columns[i]].fillna(data[data.columns[i]].mean())
            data.drop('Date', inplace=True, axis=1)
            return data

        y_pred = pd.DataFrame(filename)

        for i in range(len(filename)):
            data = dataCreator(filename[i])
            data.shape

            y = flood_classifier(filename[i], data, validating=0)

            y_pred[filename[i]] = y

        y_pred.to_csv(
            '/home/boyking/project /Rainfall_Flood/sourceCode/data/forecast'
            '/forecasted_level_of_rivers.csv',
            index=False)
        return
