import requests
import csv
import time
from datetime import date, timedelta

from key import _PRIVATE_KEY

# Variables
start_year = 2013
start_month = 12
start_date = 22
zip_code = 60604

begin_date = date(start_year, start_month, start_date)
delta = timedelta(days=1)

dates = []
for i in range(500):
    dates.append(begin_date - i * delta)

finish_date = dates[-1]
print("End Date : ", finish_date)

outPath = '%s-%s-%d.csv' % (begin_date, finish_date, zip_code)  # output path
api = _PRIVATE_KEY  # developer API key

# Create list of dates between start and end
print("Number of datapoints that will be written:", len(dates))

dates = [str(d).replace('-', '') for d in dates]

# Create daily url, fetch json file, write to disk
with open(outPath, 'a+', newline='') as csvfile:
    f = csv.writer(csvfile)
    f.writerow(
        ["datetime", "tempm", "tempi", "dewptm", "dewpti", "hum", "wspdm", "wspdi", "wgustm", "wgusti", "wdird",
         "wdire", "vism", "visi", "pressurem", "pressurei", "windchillm", "windchilli", "heatindexm", "heatindexi",
         "precipm", "precipi", "conds", "fog", "rain", "snow", "hail", "thunder", "tornado"])

    for i, day in enumerate(dates):
        if (i + 1) % 10 == 0:
            print("Finished processing %d requests" % (i + 1))
            print("Sleeping for 1 minute to avoid excess API calls")
            print()
            #time.sleep(65)

        r = requests.get('http://api.wunderground.com/api/' + api + '/history_' + day + '/q/' + str(zip_code) + '.json')
        data = r.json()['history']['observations']
        for elem in data:
            f.writerow([elem["utcdate"]["year"] + elem["utcdate"]["mon"] + elem["utcdate"]["mday"] + 'T' +
                        elem["utcdate"]["hour"] + elem["utcdate"]["min"], elem["tempm"], elem["tempi"], elem["dewptm"],
                        elem["dewpti"], elem["hum"], elem["wspdm"], elem["wspdi"], elem["wgustm"], elem["wgusti"],
                        elem["wdird"], elem["wdire"], elem["vism"], elem["visi"], elem["pressurem"], elem["pressurei"],
                        elem["windchillm"], elem["windchilli"], elem["heatindexm"], elem["heatindexi"], elem["precipm"],
                        elem["precipi"], elem["conds"], elem["fog"], elem["rain"], elem["snow"], elem["hail"],
                        elem["thunder"], elem["tornado"]])

print("Done")
