import pandas as pd
import numpy as np
from urllib.request import unquote

from staging import resolve_data_path, construct_data_path

basepath = "raw/yelp-reviews/reviews_60601-60606.csv"
path = resolve_data_path(basepath)
df = pd.read_csv(path, header=0, error_bad_lines=False)  # type: pd.DataFrame

df = df.rename(columns={
    'reviewContent': 'review'
})

print(df.info())
print(df.head())
print()

# store full dataset
path = construct_data_path('datasets/yelp-reviews/reviews.csv')
df.to_csv(path, header=True, index=True, encoding='utf-8', index_label='id')