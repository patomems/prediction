import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.svm import SVR
import numpy as np
import seaborn as sns
import plotly
import joblib
import warnings

warnings.filterwarnings("ignore")


def flood_classifier(filename, fd, validating=0, braek=None):
    data1 = pd.read_excel('/home/boyking/project /Rainfall_Flood/sourceCode/data/' + filename + '.xlsx')
    data1.shape
    for i in range(1, len(data1.columns)):
        data1[data1.columns[i]] = data1[data1.columns[i]].fillna(data1[data1.columns[i]].mean())
        data1.describe()
        y = data1['Flood']
        for i in range(len(y)):
            if y[i] >= 0.5:
                y[i] = 1.0

                y = pd.DataFrame(y)
                data1.drop('Flood', axis=1, inplace=True)
                data1.head()
                data1.hist(figsize=(6, 6));

                d1 = pd.DataFrame()
                d1["Day"] = data1['Date']
                d1['Months'] = data1['Date']
                d1['Year'] = data1['Date']

                data1['Date'] = pd.to_datetime(data1['Date'])

                d1["Year"] = data1.Date.dt.year
                d1["Months"] = data1.Date.dt.month
                d1["Day"] = data1.Date.dt.day

                dx = pd.DataFrame()
                dx['Date'] = data1['Date']
                dx['Discharge'] = data1['Discharge']
                dx = dx.set_index(['Date'])
                yearly = dx.resample('Y').sum()

                plt.figure(figsize=(9, 8))
                plt.xlabel('YEARS')
                plt.ylabel('Level')
                plt.title(filename + " : Year wise Trends")
                plt.plot(yearly, '--')
                plt.savefig('/home/boyking/project /Rainfall_Flood/sourceCode/static/img/flood.png')

                data1.drop('Date', inplace=True, axis=1)
                data1 = pd.concat([d1, data1], axis=1)
                data1.head(6)

                locate = 0;
                for i in range(len(data1["Day"])):
                    if data1["Day"][i] == 31 and data1["Months"][i] == 12 and data1["Year"][i] == 2015:
                        locate = i;
                        braek;

                        i = locate + 1
                        print(int(i))                               #

                        x_train = data1.iloc[0:i, :]
                        y_train = y.iloc[0:i]
                        x_test = data1.iloc[i:, :]
                        y_test = y.iloc[i:]

                        x_train.drop(labels=['Day', 'Months', 'Year'], inplace=True, axis=1)
                        x_test.drop(labels=['Day', 'Months', 'Year'], inplace=True, axis=1)

                        sm = SMOTE(random_state=2)
                        X_train_res, Y_train_res = sm.fit_sample(x_train, y_train)
                        X_train_res_shuf, y_train_res_shuf = shuffle(X_train_res, Y_train_res, random_state=0)            #res
                        x_train.shape, x_test.shape, y_train.shape, y_test.shape

                        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                        path = '/home/boyking/project /Rainfall_Flood/sourceCode/trained/' + filename + '_LDA'
                        
                        clf1 = joblib.load(path + '.pkl')
                        y_predict3 = clf1.predict(x_test)
                        print(set(y_predict3))
                        print(clf1.score(x_train, y_train))
                        print(clf1.score(x_test, y_test))
                        print(classification_report(y_test, y_predict3))
                        mae = mean_absolute_error(y_test, y_predict3)
                        print("mean_absolute_error=", mae)

                        data1.head()

                        def predicting(future_data):
                            xx = future_data
                            xx = np.array(xx)
                            xx = xx.reshape((-1, 4))
                            xx = clf1.predict(xx)
                            return xx

                        xx = predicting(fd)
                        return xx, mae
