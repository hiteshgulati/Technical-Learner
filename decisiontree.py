import pandas as pd
import numpy as np
from sklearn import tree
import os
from datetime import datetime
from datetime import timedelta
import sklearn



def symbol_to_path (symbol, data_dir = 'data'):
    return  os.path.join ( data_dir, '{}.csv'.format(symbol))

# Get data of single stock by passing symbol and dates from data directory
def get_data (symbol):
    df = pd.read_csv (symbol_to_path(symbol), na_values=['nan'], index_col=0)
    df =df.dropna()
    return df


def run():
    training_data = get_data('training_data')
    testing_data = get_data('testing_data')
    clf = tree.DecisionTreeClassifier()
    x_columns = training_data.columns.values.tolist()
    y_columns = x_columns.pop()
    print x_columns
    print y_columns
    x_training = training_data[x_columns]
    y_training = training_data[y_columns]
    x_testing = testing_data[x_columns]
    y_testing = testing_data[y_columns]
    clf = clf.fit(x_training,y_training)
    print "Score is: "
    print clf.score(x_testing,y_testing)
    print clf.feature_importances_
    HP_pred = y_testing.copy()
    MAC_pred = y_testing.copy()
    for i in range(0, len(y_testing)):
        if x_testing.iloc[i]['HP_Notrend'] == 1:
            HP_pred[i] = 0
        elif x_testing.iloc[i]['HP_Uptrend'] == 1:
            HP_pred[i]= 1
        elif x_testing.iloc[i]['HP_Downtrend'] == 1:
            HP_pred[i] = -1
        if x_testing.iloc[i]['MAC_Buy'] == 1:
            MAC_pred[i] = 1
        elif x_testing.iloc[i]['MAC_Sell'] == 1:
            MAC_pred[i]= -1
        elif x_testing.iloc[i]['MAC_Hold'] == 1:
            MAC_pred[i] = 0
    print "Learner Score: ", sklearn.metrics.f1_score(y_testing,clf.predict(x_testing))
    print "Hp score: ", sklearn.metrics.f1_score(y_testing,HP_pred)
    print "MAC score: ", sklearn.metrics.f1_score(y_testing, MAC_pred)



if __name__ == '__main__':
    run()
