---
layout: post
title: Weekly Report 3 - Feb 19-23 2018
---

# Amlaan

### What I did

- Experimented with Decision Trees and Random Forrests from Scikit
- Experimented with simple XGBoost
- Explored ways to play around with text data using NLTK

### What I plan to do next week

- Scrape Yelp and Wunderground (weather) data
- Use GridSearchCV, RandomizedSearchCV for parameter tuning for various ML models
- Start on data visualization libraries

### Blocks or impediments to the plan for next week

- The main issue will be scraping the Yelp and Wunderground (weather) data. Cleaning the data will also require some clever tricks to avoid just dropping rows with null or invalid values.

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
- Scraped the data sets I wanted to gather.
- Discussed more pipeline changes which seemed redundant after last week.
- Tried out some visualization frameworks anf libraries like JS Charts, High Charts and learnt a little bit of d3.js to know what it is about.

### What I plan to do next week
- Clean the scraped data and integrate it with our current directory structure
- Try to establish the connections which were discussed by us between datasets and see if they are actually feasible. This will allow us to see any holes in our ideas as well as asymmetry between datasets schemas.
- Discuss and implement atleast one Machine learning approach in addition to the other teammates efforts.

### Blocks or impediments to the plan for next week
- None other than usual snags in scraping from different sites according to their HTML structure and heirarchy.
- Long tutorials for some simple concepts took some of my time which could have been done sooner with a better choice of tutorial to follow. No technical difficulty yet.

# Christopher

### What I did
- Updated project structure and core modules that will be built later
- Looked at Census.Gov and CityofChicago Datasets and work related to Demographics information.
- Tried some tutorials on data extraction.

### What I plan to do next week
- Apply what has been understood for the data extraction.
- Work with the team in scrapping the yelp data as per the specifications provided.
- Learn more on visualization.

### Blocks or impediments to the plan for next week
- Nothing unprecedented, like pointed above the integration of data could need some extra effort.

