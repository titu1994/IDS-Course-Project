---
layout: post
title: Weekly Report 3 - Feb 19-23 2018
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

- Improved utility scripts to cache the sentiment dataset and preprocess the text, tokenize and tf-idf normalize for Scikit Learn Machine Learning Models
- Setup oversampling techniques such as SMOTE for better sentiment analysis balance.
- Setup training scripts for sentiment analysis.
- Trained a rough sentiment embedding using FastText model to improve recognition of negative sentiment.
- Trained deep learning models for Keras sentiment analysis:
  - Attention LSTM
  - Multiplicative LSTM
  - Nested LSTM
  - Neural Architecture Search RNN Cell
  - Minimal RNN Cell
  - CNN
  - FastText
  - MLP
- Try to remove extreme positive bias from the following models:
  - Logistic Regression
  - Decision Trees
  - Random Forest
- Scraped data from Wunderground API for Weather

### What I plan to do next week

- Setup Scrapy to scrape the details for the Yelp dataset for zip code 60604.
- Find some way to improve from the 60% f1 score for Logistic regression
- Find some way to improve the embeddings for Deep Learning models.

### Blocks or impediments to the plan for next week

- No blocks for now until integration begins.

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

