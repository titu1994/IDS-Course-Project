
# coding: utf-8
import itertools
import csv
import pandas as pd
import sys  
import os


def main():

    food_inspection = sys.argv[1]
    business_licences = sys.argv[2]
    
    if not os.path.isfile(food_inspection):
        print("File path {} does not exist. Exiting...".format(food_inspection))
        sys.exit()

    if not os.path.isfile(business_licences):
        print("File path {} does not exist. Exiting...".format(business_licences))
        sys.exit()

    csv.register_dialect('myDialect',delimiter=',', skipinitialspace=True)
# Open the CSV Files that are data input 
    with open(food_inspection, 'r') as csvFile:
        reader = csv.DictReader(csvFile, dialect='myDialect')
        data ={}
        for row in reader: 
            for header,value in row.items():
                try:
                    value.strip()
                
                    data[header].append(value)
                
                except KeyError:
                    data[header]=[value]

    with open(business_licences, 'r') as csvFile:
        reader = csv.DictReader(csvFile, dialect='myDialect')
        data2 ={}
        for row in reader: 
            for header,value in row.items():
                try:
                    value.strip()
                
                    data2[header].append(value)
                
                except KeyError:
                    data2[header]=[value]
                
# **************************Start of Data Cleaning and Pre Processing*****************#


# Filter the needed Columns

    res_names = data['AKA Name']
    results = data['Results']
    ins_date = data['Inspection Date']
    ins_year = pd.DatetimeIndex(ins_date)
    lids=data['License #']
    addresses=data['Address']

    year_lic = data2['DATE ISSUED']
    year_lic1 = pd.DatetimeIndex(year_lic)
    lic= data2['LICENSE NUMBER']      

    list1=[]
    list2=[]

# Fetch Restaurant for which the Inspection has failed

    for x,y,z,lid,ad,idt in zip(results,res_names,ins_year.year,lids,addresses,ins_date):
            if x == "Fail":
                if y != "":
                
                    list1.append(y)
                    list1.append(z)
                    list1.append(lid)
                    list1.append(ad)
                    list1.append(idt)
                    list2.append(list1)
                    list1=[]
    list2.sort()

# Remove Duplicates
    list2= list(list2 for list2,_ in itertools.groupby(list2))

# Just fetch the most recent record for which the inspection has failed 

    list1 =[]
    i=0
    while i < (len(list2)-1):
               if list2[i][0] != list2[i+1][0] and list2[i][1] != list2[i+1][1] :
                  list1.append(list2[i])
           
               i+=1

    

    d={}

    for x,y in zip(year_lic1.year,lic):
       
            try : 
                if d[y] < x:
                    d[y] = x
            except KeyError:
                    d[y] = x
                      
########################End of Data Cleaning and Pre Processing###################    

# Actual logic to just fetch restaurants based on Licence ID

    i=0
    data=[['Restaurant Name', 'Address', 'Failed inspection on', 'Alive for x years']]

    while i < (len(list1)-1):
        for k, v in d.items():
            if list1[i][2] == k and  v > list1[i][1] :
                    dout = []
                    active_years = v - list1[i][1]
                    dout.append(list1[i][0])
                    dout.append(list1[i][3])
                    dout.append(list1[i][4])
                    dout.append(active_years)
                    data.append(dout)
        i+=1
# Finally write output in csv file

    df = pd.DataFrame(data)
    df.to_csv('Q8.csv', index=False, header=False)

if __name__ == '__main__':  
       main()






