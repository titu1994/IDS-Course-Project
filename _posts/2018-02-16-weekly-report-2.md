---
layout: post
title: Weekly Report 2 - Feb 12-16 2018
---

# Amlaan

### What I did

- Curated three Chicago related datasets:
    - Employee Payroll (cookcountyil.gov)
    - 911-Finance (illinois.gov)
    - School incident reports (isbe.net)
- Wrote scripts to clean, normalize, and visualize said datasets using Numpy, Pandas, and Seaborn (barplot, heatmap, boxplot, kdeplot, violinplot).
- Experimented with Scikit library's Decision trees, SVM, and ExtraTreesClassifier libraries.
- Ran a test script for XGBoost on dummy dataset for possible regression/classification tasks in later stages.

### What I plan to do next week

- Work on curating datasets specified in "Data Extraction" portion of the project description
- Start integration of different datasets and normalize the data
- Visualize data in various formats to check for feature importance

### Blocks or impediments to the plan for next week

- Data integration would be the biggest problem. This also includes combining similar columns, changing data formats so they match, and finally start answering the mentioned queries.

# Somshubra

### What I did

- Setup utility scripts to load the sentiment dataset and preprocess the text, tokenize and tf-idf normalize for Scikit Learn Machine Learning Models
- Setup Scikit-Learn utility script to make management of various machine learning model training and evaluation much easier.
- Setup Keras utility script to support training of various deep learning model uniformly.
- Generated the Embedding matrix required for deep learning models using Glove 840B words embedding available.
- Added deep learning layers for Keras:
  - Attention LSTM
  - Multiplicative LSTM
  - Nested LSTM
  - Neural Architecture Search RNN Cell
  - Minimal RNN Cell
- Created training and evaluation scripts for training and evaluating below ML Models on Sentiment dataset:
  - Logistic Regression
  - Decision Trees
  - Random Forest
- Created training and evaluation scripts for training and evaluating below DL Models on Sentiment dataset:
  - FastText (from FAIR)
  - CNN
- Scraped data from Wunderground API for Weather

### What I plan to do next week

- Finish scraping and building the weather dataset by scraping more data for previous years.
- Train more Deep Learning models from the RNN branch and attempt to improve performance of ML models
- Train Linear SVM and other linear ML models on the Sentiment dataset. Perhaps try XGBoost and LightGBM if time permits.

### Blocks or impediments to the plan for next week

- Next week, we should begin integrating all the datasets. Need everyone to gather their datasets and clean them up to prepare for integration.

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
- Referred UIC Library Resource Databases and We Search and presented below datasets:.
	-Demographics and Socioeconomic Characteristics
	-Cook County Statistics 
	-Businesses in Chicago
	-Real Estate Chicago
- Finalized the data and drafted the data related to above parts for Report1.
- Integrated and tabulated the data source links and related attributes as part of Report1. 
- Had a team meeting and spent time in understanding work that has to be done and what is being done by the team.

### What I plan to do next week
- Understanding the next phase of Project Requirement.
- Look at Census.Gov and CityofChicago Datasets and work related to Demographics information.
- Learn more on Data Extraction Utilities.

### Blocks or impediments to the plan for next week
- Nothing unprecedented, except that there will time spent in understand the process. 
