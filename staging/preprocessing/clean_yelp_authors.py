import pandas as pd
import numpy as np

from staging import construct_data_path, resolve_data_path

data_path = resolve_data_path('raw/yelp/yelp_users.json')
df = pd.read_json(data_path)  # type: pd.DataFrame

print(df.info())

data_path = construct_data_path('datasets/yelp/author.csv')
df.to_csv(data_path, encoding='utf-8', index=True, index_label='authorID')
