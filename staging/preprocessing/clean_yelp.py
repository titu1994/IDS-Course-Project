import pandas as pd
import numpy as np
import re

import sys
sys.path.insert(0, "..")

from staging import construct_data_path, resolve_data_path

basepath = "raw/yelp/"
pin_codes = [60604, 60612]
# Ref: https://stackoverflow.com/questions/2779453/python-strip-everything-but-spaces-and-alphanumeric
clean_name_pattern = re.compile(r'([^\s\w]|_)+')

#fmt1 = basepath + "yelp_%d.json"
#fmt2 = basepath + "yelp_%d_bar.json"
#fmt3 = basepath + "yelp_full_%d.json"
#fmt4 = basepath + "yelp_full_%d_bar.json"

# restaurants_partial = []
# restaurants_full = []
# bar_partial = []
# bar_full = []

# for pincode in pin_codes:
#     restaurants_partial.append(resolve_data_path(fmt1 % pincode))
#     bar_partial.append(resolve_data_path(fmt2 % pincode))
#     restaurants_full.append(resolve_data_path(fmt3 % pincode))
#     bar_partial.append(resolve_data_path(fmt4 % pincode))

def clean_name(df):
    name = df['Name']
    name = clean_name_pattern.sub('', name)
    words = name.split()

    clean_name = []
    for word in words:
        word = clean_name_pattern.sub('', word)
        clean_name.append(word)

    clean_name = ' '.join(clean_name)
    return clean_name

def clean_reviews(review):
    try:
        x = int(review)
    except:
        x = int(''.join([ch for ch in review if ch.isdigit()]))
    return x

price_map = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4, 'x': 0}

def clean_price(df):
    val = df['Price']

    if len(val) < 1:
        val = 'x'
    else:
        val = val[0]

    val = price_map[val]
    return val

def get_pin_code(df):
    address = df['Address']
    pincode = address[-5:]
    if len(pincode) < 1:
        return np.nan

    try:
        pincode = int(pincode)
    except:
        return np.nan

    return pincode

dataframe_list = []


def load_dataframe(file):
    df = pd.read_json(file)  # type: pd.DataFrame
    df['Address'] = df['Address'].astype(np.str)
    df['Name'] = df.apply(clean_name, axis=1).astype(np.str)
    df['Category'] = df['Category'].astype(np.str)
    df['Price'] = df.apply(clean_price, axis=1)  #
    df['ReviewCount'] = df['ReviewCount'].apply(clean_reviews)
    #df['Alcohol'] = alcohol
    df['Pincode'] = df.apply(get_pin_code, axis=1)

    df = df.dropna()

    pins = df['Pincode'].isin(pin_codes)
    df['Pincode'] = df.Pincode.loc[pins == True]
    df = df.dropna()

    df['Pincode'] = df['Pincode'].astype(np.int)

    print(df.info())
    print(df.head(5))

    return df

dataset_path = resolve_data_path('raw/yelp/yelp_data.json')
df = load_dataframe(dataset_path)

# for file in restaurants_partial:
#     df = load_dataframe(file, alcohol=0)
#     dataframe_list.append(df)
#
# for file in restaurants_full:
#     df = load_dataframe(file, alcohol=0)
#     dataframe_list.append(df)
#
# for file in bar_partial:
#     df = load_dataframe(file, alcohol=1)
#     dataframe_list.append(df)
#
# for file in bar_full:
#     df = load_dataframe(file, alcohol=1)
#     dataframe_list.append(df)
#
#df = pd.concat(dataframe_list)  # type: pd.DataFrame

print()
print(df.info())

df = df.drop_duplicates(subset=['BusinessID'])

print()
print(df.info())
print(df.head())

save_path = construct_data_path('raw/yelp/cleaned_yelp_ratings.csv')
df.to_csv(save_path, index=True, index_label='id', header=True, encoding='utf-8')

names = df['Name'].values
print('\n\n,', names.tolist())

urls = df['YelpURL'].values
print("\n\n", urls.tolist())