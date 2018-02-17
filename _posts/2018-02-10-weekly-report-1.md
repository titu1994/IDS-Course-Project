---
layout: post
title: Weekly Report 1 - Feb 05-09 2018
---

# Amlaan

### What I did

- Explored datasets online; especially on kaggle.com.
- Curated three Chicago related datasets:
    - Crimes in Chicago (~1.4m samples)
    - Chicago Restaurant Inspections (~154k samples)
    - Chicago - Citywide Payroll Data (~32k samples)
- Wrote scripts to clean, handle missing data, and store Crimes and Restaurant Inspection datasets (fully usable for ML models)
- Created Wunderground API key to get weather data when zipcode is available

### What I plan to do next week

- Explore and clean more datasets
- Start out on a strategy to integrate various available and usable datasets with entity resolution techniques
- Try out Decision trees and Random Forrests from Scikit library

### Blocks or impediments to the plan for next week

- None for now. The biggest issue would be data integration and handling conflicting and empty data that arises when columns are integrated.

# Somshubra

### What I did

- Setup project structure and core modules that will be built later
- Setup scrapers to scrape yelp.com for restaurants and bars around Chicago from the address 60601 - 60607. Scraped roughly 550~ restaurants + bars and some useful data.
- Setup scrapers to scrape yelp.com/bi for reviews based on the names of restaurants from the above dataset. Scraped roughly 50,000 reviews of nearly 280 restaurants + bars.
- Obtained Yelp Dataset for nearly 5.2 million raw reviews of various restaurants from all over the U.S. We plan to use this to supplement the final classifier to be more generalized and learn to predict sentiment based on just the text description alone.
- Setup script to call the Wunderground API to obtain nearly 4 years worth of weather data for a pin codes 60601 & 60605.
- Wrote scripts to clean and preprocess the above scraped datasets and utils to manipulate the data.

### What I plan to do next week

- Finish scraping and building the weather dataset by scraping more data for previous years.
- Setup basic preprocessing for cleaning the text reviews
- Setup basic ML baselines for Sentiment Analysis of the reviews, if time permits

### Blocks or impediments to the plan for next week

- None for the foreseeable future. However, next week will be data integration. So we need to prepare all of the other datasets asap.

# Debojit

### What I did
- Imported and ran the scripts to collect data. Updated myself with the structure and workflow.
- Surfed the internet to decided on a few datasets.
- Designed a initial pipeline for the project and imparted it to teammates

### What I plan to do next week
- Scrape data from some sites and download a dataset I have decided on.
- Clean the scraped data into the commonly decided format.
- Discuss with teammates the required duties and responsibilities.

### Blocks or impediments to the plan for next week
- None so far, except a few conditions to be decided while cleaning the data and integrating it with the other three.

# Christopher

### What I did
- Imported project structure and core modules that will be built later
- As part of data discovery; for now obtained the public bike data
- Spent time in understanding work that has to be done and what is being done by the team.

### What I plan to do next week
- Planning on getting the data in the health sector and relate it to the existing requirements
- Must finalize the data and draft the first report.
- Learn more on visualization, for that is where I look forward to contribute primarily.

### Blocks or impediments to the plan for next week
- Nothing unprecedented, like pointed above the integration of data could need some extra effort.

