import pandas as pd
import numpy as np

files = [
    '2008-07-05-2007-02-22-60604.csv',
    '2009-11-16-2008-07-05-60604.csv',
    '2011-03-31-2009-11-17-60604.csv',
    '2012-08-01-2011-03-21-60604.csv',
    '2013-12-22-2012-08-10-60604.csv',
    '2015-05-06-2013-12-23-60604.csv',
    '2016-09-18-2015-05-08-60604.csv',
    '2018-01-31-2016-09-19-60604.csv',
]

df_list = []  # type: pd.DataFrame
for df_path in files:
    df = pd.read_csv(df_path, header=0, encoding='utf-8')  # type: pd.DataFrame
    df.dropna(inplace=True)
    df = df[::-1]
    df_list.append(df)

df = pd.concat(df_list, axis=0)

print(df.info())

df.to_csv("weather.csv", encoding='utf-8', index=True, index_label='id')

