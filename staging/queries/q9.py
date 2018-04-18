# coding: utf-8

import csv
import pandas as pd
import numpy as np
import itertools
from bs4 import BeautifulSoup
import requests
import re

business_lic_file = 'Business_Licenses_Liq.csv'

d = {}
B = {}
with open(business_lic_file) as i_file:
    reader = csv.reader(i_file)
    next(reader, None)
    for row in reader:
        id = row[0].split('-')[0]

    try:
        if row[0] != "":
            l = []
            l.append(row[31])
            l.append(row[32])
            d[id] = l

    except KeyError:
        d[y] = x

for k, v in d.items():

    lt = v[1]
    ln = v[0]
    l = []

    url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates?x=%s&y=%s&benchmark=4&vintage=4" % (lt, ln)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    blk = soup.findAll(text=re.compile('GEOID'))
    try:
        block = blk[2][12:18] + '_' + blk[2][18:]
    except IndexError:
        continue

    if block in B:
        l = B[block]
        l.append(k)
        B[block] = l
    else:
        l.append(k)
        B[block] = l


data = []
l = []
for k, v in B.items():
    if k != "_":
        l.append(k)
        l.append(len(v))
        data.append(l)
        l = []

# Finally write output in csv file
data.sort()
data[0] = ['Block_code', '#Businesses with liquor licenses', '#Crimes', '#Arrests']
df = pd.DataFrame(data)
df.to_csv('Q9.csv', index=False, header=False)


l = []
l2 = []
a_file = 'crimes.csv'
with open(a_file) as inp_file:
    reader = csv.reader(inp_file)
    next(reader, None)

    for row in reader:
        if row[14] and row[15] != '':
            l.append(row[14][:6])
            l.append(row[15][:7])
            l.append(row[7])
            l2.append(l)
            l = []

Z = {}
for k, v in d.items():
    for i in l2:

        if i[0] == v[0][:6] and i[1] == v[1][:7]:

            if k in Z:
                cnt = Z[k] + 1
                Z[k] = cnt
            else:
                cnt = 1
                Z[k] = cnt

F = {}

for k, v in Z.items():

    for key, val in B.items():
        if b_any(k in x for x in val):

            if key in F:
                F[key] = F[key] + Z[k]
            else:
                F[key] = Z[k]

for k, v in F.items():
    print(k, v)


last = []
sec = []
fin = {}

with open('Q9.csv') as f_file:
    reader = csv.reader(f_file)
    next(reader, None)
    for row in reader:
        last.append(row[0])
        sec.append(row[1])
crime = []
data = []

for x, y in zip(last, sec):
    for k, v in F.items():
        if x == k:
            crime.append(x)
            crime.append(y)
            crime.append(v)
            data.append(crime)
            crime = []

data[0] = ['Block_code', '#Businesses with liquor licenses', '#Crimes', '#Arrests']
df = pd.DataFrame(data)
df.to_csv('Q9_final.csv', index=False, header=False)
