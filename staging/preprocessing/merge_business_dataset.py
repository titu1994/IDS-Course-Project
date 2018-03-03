import pandas as pd
import numpy as np

import re

from staging import construct_data_path, resolve_data_path

#pin_codes = [60604,]

business_drop_cols = ['Accepts Android Pay',
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

business_rename_dict = {
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
        'hours': 'Hours',
    }

business_nan_to_false_cols = ['AcceptsCreditCards', 'Alcohol', 'Ambience', 'Attire', 'Caters', 'Delivery', 'GoodFor',
                 'GoodforGroups', 'GoodforKids', 'HasTV', 'NoiseLevel', 'OutdoorSeating', 'Parking', 'Takeout',
                 'TakesReservations', 'WaiterService', 'WheelchairAccessible', 'WiFi']

business_nan_to_empty_cols = ['phoneNumber', 'website']


def drop_columns(df:pd.DataFrame, col_list):
    df = df.drop(col_list, axis=1)
    return df

def rename_columns(df:pd.DataFrame, rename_dict):
    df = df.rename(columns=rename_dict)
    return df


def set_nan_to_false(df:pd.DataFrame, col_names):
    df[col_names] = df[col_names].fillna(value=0.0)
    return df

def set_nan_to_empty(df:pd.DataFrame, col_names):
    df[col_names] = df[col_names].fillna("")
    return df


dataset_path = resolve_data_path('raw/yelp/cleaned_yelp_ratings.csv')
df = pd.read_csv(dataset_path, header=0, encoding='utf-8', index_col='id')  # type: pd.DataFrame

print(df.info())
print(df.head())

businesses_path = resolve_data_path('raw/yelp/yelp_business.json')
bdf = pd.read_json(businesses_path)  # type: pd.DataFrame

bdf = drop_columns(bdf, business_drop_cols)
bdf = rename_columns(bdf, business_rename_dict)
bdf = set_nan_to_false(bdf, business_nan_to_false_cols)
bdf = set_nan_to_empty(bdf, business_nan_to_empty_cols)

print(bdf.info())
print(bdf.head())

# merge the two datasets
merged_df = pd.merge(df, bdf, left_on='BusinessID', right_on='restaurantID')  # type: pd.DataFrame

merged_drop_cols = ['BusinessID', 'Phone', 'YelpURL']

merged_rename_dict = {
    'Price': 'PriceRange',
    'website': 'webSite',
    'Neighbourhood': 'location',
}

merged_df = drop_columns(merged_df, merged_drop_cols)
merged_df = rename_columns(merged_df, merged_rename_dict)

print(merged_df.info())

data_path = construct_data_path('datasets/yelp/restaurant.csv')
merged_df.to_csv(data_path, index_label='id', encoding='utf-8')

