import censusgeocode as cg
import pickle
import time
import os
import pandas as pd
import numpy as np
import csv
from sklearn import tree, linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib


def read_dataset():
    if not os.path.exists('crimes.p'):
        crimes_init = pd.read_csv('crimes.csv')
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


def read_weather_dataset():
    if not os.path.exists('weather.p'):
        weather = pd.read_csv('weather.csv')
        dates = ['201406', '201407', '201408', '201409', '201506', '201507', '201508', '201509', '201606', '201607', '201608', '201609', '201706', '201707', '201708', '201709']
        weather_pattern = '|'.join(dates)
        weather = weather[weather['datetime'].str.contains(weather_pattern)] #20170113 check for dates
        pickle.dump(weather, open('weather.p', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Weather pickle dump found!')
        start = time.time()
        weather = pickle.load(open('weather.p', 'r'))
        end = time.time()
        print('Elapsed time(weather): ' + str(end - start))
    weather_sample = weather.sample(n=10000)
    return weather_sample


def call_census_merge(crimes):
    crimes_sample = crimes.sample(n=10000)
    locs = crimes_sample[['lat', 'long']]
    locs = locs.values
    census_ids = []

    if not os.path.exists('census_id.p'):
        for idx, i in enumerate(locs):
            try:
                X = cg.coordinates(x=i[1], y=i[0])
                census_ids.append(X['2010 Census Blocks'][0]['TRACT'] + '_' + X['2010 Census Blocks'][0]['BLOCK'])
                print(str(idx) + '=' + str(X['2010 Census Blocks'][0]['TRACT'] + '_' + X['2010 Census Blocks'][0]['BLOCK']))
            except:
                census_ids.append(np.NaN)
                print('error for: ' + str(i[0]) + ', ' + str(i[1]))
        pickle.dump(census_ids, open('census_id.p', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(crimes_sample, open('crimes_sample.p', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Census pickle dump found!')
        start = time.time()
        census_ids = pickle.load(open('census_id.p', 'r'))
        crimes_sample = pickle.load(open('crimes_sample.p', 'r'))
        end = time.time()
        print('Elapsed time(crimes): ' + str(end - start))

    census_ids = pd.DataFrame(census_ids)
    # pd.merge(crimes_sample, census_ids, how='outer', left_index=True, right_index=True)
    # crimes_sample = crimes_sample.join(census_ids, how='outer')
    # crimes_sample2 = crimes_sample.merge(census_ids, how='outer', left_index=True, right_index=True)
    # crimes_sample2 = crimes_sample.join(census_ids, how='left').astype(str)
    # pd.merge(crimes_sample, census_ids, how='left', left_index=True)
    crimes_sample['census_id'] = census_ids.values
    crimes_sample.columns = ['date', 'block', 'crime_pd', 'crime_sd', 'crime_ld', 'arrest', 'beat', 'ward', 'x_cord',
                             'y_cord', 'lat', 'long', 'census_id']
    crimes_sample = crimes_sample[['census_id', 'date', 'crime_pd', 'crime_ld', 'block', 'crime_sd', 'arrest', 'beat']]
    return crimes_sample


def merge_weather_and_crimes(crimes, weather):
    print(crimes.shape)
    weather_temp = weather[['tempm']]
    crimes['tempm'] = weather_temp.values
    # crimes2 = crimes.join(weather_temp).astype(str) # last working code
    # crimes['avg_temp'] = weather['tempm'].astype(str)
    # crimes.avg_temp = crimes.avg_temp.astype(float)
    crimes.date = crimes.date.str.slice(0, 2)  # get month from date
    crimes.date = pd.to_numeric(crimes.date)  # convert month to number
    return crimes


def train(dataset):
    # dataset = dataset[['date_of_occurrence', 'block', 'primary_description', 'secondary_description',
    # 'location_description', 'arrest', 'beat', 'ward']]
    dataset = dataset.select_dtypes(exclude=['number']) \
                .apply(LabelEncoder().fit_transform) \
                .join(dataset.select_dtypes(include=['number']))
    dataset = dataset.values
    X = dataset[:, [0, 1, 3, 4, 5, 6, 7]]
    y = dataset[:, [2]]
    c, r = y.shape
    y = y.reshape(c,)
    skf = StratifiedKFold(n_splits=2)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dt = tree.DecisionTreeClassifier()
        lr = linear_model.LogisticRegression()
        print('DecisionTree score: %f' % dt.fit(X_train, y_train).score(X_test, y_test))
        print('LogisticRegression score: %f' % lr.fit(X_train, y_train).score(X_test, y_test))
        joblib.dump(dt, 'decisiontree.pkl')
        joblib.dump(lr, 'logisticregression.pkl')
    return X, y


def predict(X, og):
    crime_types = og.crime_pd.unique()
    og = og.values
    if os.path.exists('decisiontree.pkl'):
        dt = joblib.load('decisiontree.pkl')
        lr = joblib.load('logisticregression.pkl')
        dt_preds = dt.predict_proba(X)
        lr_preds = lr.predict_proba(X)
        with open('q10-result.csv', 'wb') as csvfile:
            preds_writer = csv.writer(csvfile, delimiter=',')
            preds_writer.writerow(['CENSUS BLOCK', 'MONTH', 'AVG TEMPERATURE', 'TYPE OF ROBBERY', 'PROBABILITY'])
            for i in range(dt_preds.shape[0]):
                for j in range(len(crime_types)):
                    preds_writer.writerow([og[i][0], og[i][1], og[i][8], crime_types[j], lr_preds[i][j]])
    else:
        print('Model not trained, train it first')


if __name__ == '__main__':
    crimes = read_dataset()
    weather = read_weather_dataset()
    merge1 = call_census_merge(crimes)
    merge2 = merge_weather_and_crimes(merge1, weather)
    X, y = train(merge2)
    predict(X, merge2)