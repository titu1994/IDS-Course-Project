import sys
import pandas as pd
import numpy as np
import usaddress
import os
import pickle
import time
import csv
from math import radians, cos, sin, asin, sqrt
sys.path.insert(0, "..")

def read_datasets(args):
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

    if not os.path.exists('bl.p'):
        business_licenses_init = pd.read_csv(args[2])
        business_licenses = business_licenses_init[['LEGAL NAME', 'DOING BUSINESS AS NAME', 'ADDRESS', 'WARD',
                                                    'PRECINCT', 'WARD PRECINCT', 'POLICE DISTRICT',
                                                    'LICENSE DESCRIPTION', 'BUSINESS ACTIVITY',
                                                    'LATITUDE', 'LONGITUDE']]
        business_licenses.columns = ['legal_name', 'dba', 'address', 'ward', 'precinct', 'ward_precinct',
                                     'police_district', 'license_desc', 'business_act', 'lat', 'long']
        pickle.dump(business_licenses, open('bl.p', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Business license pickle dump found!')
        start = time.time()
        business_licenses = pickle.load(open('bl.p', 'r'))
        end = time.time()
        print('Elapsed time(bl): ' + str(end - start))

    return crimes, business_licenses

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    
    Ref: https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-
    the-distance-between-two-latitude-longitude-points
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km

def determine_type(row):
    name = row['name'].upper()
    if 'RESTAURANT' in name:
        return 'c'
    elif 'SCHOOL' in name:
        return 'b'
    else:
        return 'a'

def determine_alcohol(row):
    t = row['business_type']
    if t == 'a':
        return 'Y'
    elif t == 'c':
        return 'Y'
    else:
        return 'N'

def determine_tobacco(row):
    t = row['business_type']
    if t == 'a':
        return 'Y'
    else:
        return 'N'
    
def join_list(list):
    return ' '.join(list[-3:])

def clean_datasets(crimes, bl):
    #print(crimes.columns)
    #print(bl.columns)
    # print(bl.head)
    list_of_filters = ['restaurant', 'Restaurant', 'school', 'School', 'grocery', 'Grocery', 'RESTAURANT',
                      'SCHOOL', 'GROCERY']
    pattern = '|'.join(list_of_filters)
    bl = bl[bl['legal_name'].str.contains(pattern)] # check if business is rest, school, or grocery
    bl = bl.drop_duplicates(['legal_name', 'dba', 'address'], keep='first')
    # Find distance between crimes and businesses
    crimes = crimes.values
    bl = bl.values
    
    #print(crimes.shape)
    #print(bl.shape)
    
    if not os.path.exists('combo.csv'):
        crimes_bl_combo = []
        with open('combo.csv', 'wb') as csvfile:
            c_writer = csv.writer(csvfile, delimiter=',')
            c_writer.writerow(['date', 'legal_name', 'dba', 'bl_address', 'crime_block', 'bl_activity', 'crime_pd', 'crime_ld', 'arrest', 'license_desc', 'distance'])
            for i in range(bl.shape[0]):
                for j in range(crimes.shape[0]):
                    dist = haversine(crimes[j][11], crimes[j][10], bl[i][10], bl[i][9])
                    if dist < 0.32: # assume 3 blocks = 0.32 km radius
                        c_writer.writerow([crimes[j][0], bl[i][0], bl[i][1], bl[i][2], crimes[j][1], bl[i][8], crimes[j][2], crimes[j][4], crimes[j][5], bl[i][7], dist])
    else:
        print('combo.csv found!')
        start = time.time()
        crimes_bl_combo = []
        with open('combo.csv', 'rb') as csvfile:
            c_reader = csv.reader(csvfile, delimiter=',')
            for row in c_reader:
                crimes_bl_combo.append(row)
        end = time.time()
        print('Elapsed time(combo): ' + str(end - start))
    
    crimes_bl = pd.DataFrame(crimes_bl_combo) # convert list to dataframe
    crimes_bl.columns = ['year', 'name', 'business_type', 'bl_address', 'crime_block', 'bl_activity', 'crime_pd', 'crime_ld', 'arrest', 'license_desc', 'distance']
    crimes_bl = crimes_bl.iloc[1:] # drop column name row
    crimes_bl.year = crimes_bl.year.str.slice(6, 10) # slice year from date
    crimes_bl['business_type'] = crimes_bl.apply(lambda row: determine_type(row), axis=1) # determine type and put into dba
    crimes_bl = crimes_bl.drop(['crime_block'], axis=1)
    aggregate_df = crimes_bl.groupby(['year', 'name', 'business_type', 'bl_address', 'crime_pd', 'crime_ld', 'arrest']).size().reset_index().rename(columns={0:'count'})
    #aggregate_name = aggregate_df.groupby(['name']).size().reset_index().rename(columns={0:'count'})
    aggregate_arrest = aggregate_df.groupby(['name', 'arrest']).size().reset_index().rename(columns={0:'count'})
    aggregate_new = pd.merge(aggregate_df, aggregate_arrest, on='name')
    # aggregate_new.drop(aggregate_new[aggregate_new.arrest == 'N'].index, inplace=True)
    aggregate_new.drop(['arrest_x'], axis=1)
    aggregate_new['has_alcohol_license'] = aggregate_new.apply(lambda row: determine_alcohol(row), axis=1)
    aggregate_new['has_tobacco_license'] = aggregate_new.apply(lambda row: determine_tobacco(row), axis=1)
    aggregate_new = aggregate_new[['year', 'business_type', 'name', 'bl_address', 'has_tobacco_license', 'has_alcohol_license', 'crime_pd', 'count_x', 'count_y', 'crime_ld']]
    aggregate_new['crime_ld'] = aggregate_new['count_x']
    aggregate_new.columns = ['Year', 'Business Type (a,b,c)', 'Business Name', 'Address', 'Has Tobacco License', 'Has Liquor License', 'Crime Type', '#Crimes', '#Arrests', '#On Premises']
    print(aggregate_new.shape)
    # print(aggregate_new.head(3))
    print(aggregate_df.shape)
    aggregate_new.to_csv('q1-result.csv', sep=',')
    #print(crimes_bl.head(2))
    print(crimes_bl.shape)
    
if __name__ == "__main__":
    # Year, Business Type (a,b,c), Business Name, Address, 
    # Has Tobacco License, Has Liquor License, Crime Type, #Crimes, #Arrests, #On Premises
    args = sys.argv
    print(args)
    crimes, business_licenses = read_datasets(args)
    clean_datasets(crimes, business_licenses)