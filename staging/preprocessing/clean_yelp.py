import pandas as pd
import numpy as np

from staging.utils.generic_utils import resolve_data_path, construct_data_path

basepath = "raw/yelp/"
pin_codes = list(range(60601, 60608, 1))

fmt1 = basepath + "yelp_%d.json"
fmt2 = basepath + "yelp_%d_bar.json"
fmt3 = basepath + "yelp_full_%d.json"
fmt4 = basepath + "yelp_full_%d_bar.json"

restaurants_partial = []
restaurants_full = []
bar_partial = []
bar_full = []

for pincode in pin_codes:
    restaurants_partial.append(resolve_data_path(fmt1 % pincode))
    bar_partial.append(resolve_data_path(fmt2 % pincode))
    restaurants_full.append(resolve_data_path(fmt3 % pincode))
    bar_partial.append(resolve_data_path(fmt4 % pincode))

def clean_reviews(review):
    try:
        x = int(review)
    except:
        x = int(''.join([ch for ch in review if ch.isdigit()]))
    return x

price_map = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4, 'x': 0}

def clean_price(df):
    val = df['Price']

    if type(val) == list:
        if len(val) < 1:
            val = 'x'
        else:
            val = val[0]

    val = price_map[val]
    return val

def get_pin_code(df):
    address = df['Address']
    pincode = address[-5:]
    return pincode

dataframe_list = []


def load_dataframe(file, alcohol=0):
    df = pd.read_json(file)  # type: pd.DataFrame
    df['Address'] = df['Address'].astype(np.str)
    df['Category'] = df['Category'].astype(np.str)
    df['Price'] = df.apply(clean_price, axis=1)  #
    df['Reviews'] = df['Reviews'].apply(clean_reviews)
    df['Alcohol'] = alcohol
    df['Pincode'] = df.apply(get_pin_code, axis=1)

    print(df.info())
    print(df.head(5))

    return df


for file in restaurants_partial:
    df = load_dataframe(file, alcohol=0)
    dataframe_list.append(df)

for file in restaurants_full:
    df = load_dataframe(file, alcohol=0)
    dataframe_list.append(df)

for file in bar_partial:
    df = load_dataframe(file, alcohol=1)
    dataframe_list.append(df)

for file in bar_full:
    df = load_dataframe(file, alcohol=1)
    dataframe_list.append(df)

df = pd.concat(dataframe_list)  # type: pd.DataFrame

print()
print(df.info())

df = df.drop_duplicates(subset=['Name'])

print()
print(df.info())
print(df.head())

save_path = construct_data_path('datasets/yelp_ratings.csv')
df.to_csv(save_path, index=True, index_label='id', header=True, encoding='utf-8')
