import pandas as pd
import numpy as np

import re

from staging import construct_data_path, resolve_data_path

#pin_codes = [60604,]

def drop_columns(df:pd.DataFrame):
    col_list = ['Accepts Android Pay',
                'Accepts Apple Pay',
                'Accepts Bitcoin',
                'Best Nights',
                'Bike Parking',
                'Dogs Allowed',
                'Drive-Thru',
                'Gender Neutral Restrooms',
                'Good For Dancing',
                'Has Pool Table',
                'Music',
                ]

    df = df.drop(col_list, axis=1)
    return df

def rename_columns(df:pd.DataFrame):
    rename_dict = {
        'Accepts Credit Cards': 'AcceptsCreditCards',
        'Good For': 'GoodFor',
        'Good for Groups': 'GoodforGroups',
        'Good for Kids': 'GoodforKids',
        'Has TV': 'HasTV',
        'Noise Level': 'NoiseLevel',
        'Outdoor Seating': 'OutdoorSeating',
        'Take-out': 'Takeout',
        'Takes Reservations': 'TakesReservations',
        'Waiter Service': 'WaiterService',
        'Wheelchair Accessible': 'WheelchairAccessible',
        'Wi-Fi': 'WiFi',
    }

    df = df.rename(columns=rename_dict)
    return df


def set_nan_to_false(df:pd.DataFrame):
    col_names = ['AcceptsCreditCards', 'Alcohol', 'Ambience', 'Attire', 'Caters', 'Delivery', 'GoodFor',
                 'GoodforGroups', 'GoodforKids', 'HasTV', 'NoiseLevel', 'OutdoorSeating', 'Parking', 'Takeout',
                 'TakesReservations', 'WaiterService', 'WheelchairAccessible', 'WiFi']

    df[col_names] = df[col_names].fillna(value=0.0)
    return df



dataset_path = resolve_data_path('raw/yelp/cleaned_yelp_ratings.csv')
df = pd.read_csv(dataset_path, header=0, encoding='utf-8')  # type: pd.DataFrame

#print(df.info())

businesses_path = resolve_data_path('raw/yelp/yelp_business.json')
bdf = pd.read_json(businesses_path)  # type: pd.DataFrame

bdf = drop_columns(bdf)
bdf = rename_columns(bdf)
bdf = set_nan_to_false(bdf)

print(bdf.columns.tolist())

print(bdf.info())

