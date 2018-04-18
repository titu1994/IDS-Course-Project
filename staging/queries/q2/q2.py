import sys
import pandas as pd
import numpy as np
import os
import time
import pickle
import csv
from datetime import datetime
from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

def read_dataset(args):
    if not os.path.exists('crimes.p'):
        crimes_init = pd.read_csv(args[1])
        crimes_init.columns = ['case#', 'date_of_occurrence', 'block', 'iucr', 'primary_description',
                               'secondary_description', 'location_description', 'arrest', 'domestic', 'beat', 'ward',
                               'fbi_cd', 'x_cord', 'y_cord', 'lat', 'long', 'loc']
        crimes = crimes_init[['date_of_occurrence', 'block', 'primary_description', 'secondary_description',
                              'location_description', 'arrest', 'beat', 'ward', 'x_cord', 'y_cord', 'lat', 'long']]
        pickle.dump(crimes, open('crimes.p', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Crimes pickle dump found!')
        start = time.time()
        crimes = pickle.load(open('crimes.p', 'r'))
        end = time.time()
        print('Elapsed time(crimes): ' + str(end - start))
    return crimes

def join_list(list):
    return ' '.join(list[-2:])
    
def filter_dataset(crimes):
    crimes.date_of_occurrence = crimes.date_of_occurrence.str.slice(0, 2) # get month from date
    crimes.date_of_occurrence = pd.to_numeric(crimes.date_of_occurrence) # convert month to number
    
    crimes.block = crimes.block.str.split(' ')
    crimes.block = crimes.block.apply(join_list)
    print('number of unique addresses: ' + str(crimes.block.nunique()))
    print('number of unique dates: ' + str(crimes.date_of_occurrence.nunique()))
    print('number of unique crime types: ' + str(crimes.primary_description.nunique()))
    print('shape: ' + str(crimes.shape))
    return crimes

def train(dataset):
    #counts = dataset['primary_description'].value_counts()
    #dataset = dataset[dataset['primary_description'].isin(counts[counts > 1000].index)]
    dataset = dataset[['date_of_occurrence', 'block', 'primary_description', 'secondary_description', 
                       'location_description', 'arrest', 'beat', 'ward']]
    dataset2 = dataset.select_dtypes(exclude=['number']) \
                .apply(LabelEncoder().fit_transform) \
                .join(dataset.select_dtypes(include=['number']))
    dataset2 = dataset2.values
    X = dataset2[:, [0, 2, 3, 4, 5]]
    y = dataset2[:, [1]]
    c, r = y.shape
    y = y.reshape(c,)
    skf = StratifiedKFold(n_splits=2)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        lr = linear_model.LogisticRegression(max_iter=100)
        dt = tree.DecisionTreeClassifier()
        print('DecisionTree score: %f' % dt.fit(X_train, y_train).score(X_test, y_test))
        print('LogisticRegression score: %f' % lr.fit(X_train, y_train).score(X_test, y_test))
        print('RandomForest score: %f' % rfc.fit(X_train, y_train).score(X_test, y_test))
        joblib.dump(dt, 'decisiontree.pkl')
        joblib.dump(lr, 'logisticregression.pkl')
        joblib.dump(rfc, 'randomforest.pkl')
    return X, y

def predict(X, y, og):
    og = og.values
    if os.path.exists('decisiontree.pkl'):
        dt = joblib.load('decisiontree.pkl')
        lr = joblib.load('logisticregression.pkl')
        rfc = joblib.load('randomforest.pkl')
        dt_preds = dt.predict_proba(X)
        lr_preds = lr.predict_proba(X)
        rfc_preds = rfc.predict_proba(X)
        crime_types = crimes.primary_description.unique()
        with open('q2-result.csv', 'wb') as csvfile:
            preds_writer = csv.writer(csvfile, delimiter=',')
            preds_writer.writerow(['ADDRESS', 'CRIME TYPE', 'TECHNIQUE', 'PROBABILITY'])
            for i in range(dt_preds.shape[0]):
                for j in range(len(crime_types)):
                    preds_writer.writerow([og[i][1], crime_types[j], 'DECISION TREE', dt_preds[i][j]])
                    preds_writer.writerow([og[i][1], crime_types[j], 'LOGISTIC REGRESSION', lr_preds[i][j]])
                    preds_writer.writerow([og[i][1], crime_types[j], 'RANDOM FOREST', rfc_preds[i][j]])
    else:
        print('Models not trained, train them first')

if __name__ == "__main__":
    args = sys.argv
    crimes_1 = read_dataset(args)
    crimes_temp = crimes_1.copy()
    crimes = filter_dataset(crimes_1)
    X, y = train(crimes)
    predict(X, y, crimes_temp)