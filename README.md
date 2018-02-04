# IDS-Course-Project

## Deliverables

| Date  | Task to Deliver  |
| ------|:----------------:|
| 2/08	| Team names, team members, website URL and git repository URLs |
| 2/15  | Report 1 - Preliminary report containing results of discovery (1000 ± 50 words) |
| 3/01	| Report 2 - Overview of technical approach and high-level design (1000 ± 50 words) |
| 3/29	| Manual reference datasets |
| 4/12	| Fully functional application, a README.txt containing the information about how to run your code, and a file containing data results |
| 4/26	| Website and Video. Slides for in-class presentation |
| 5/03	| Report 3 - Final project report describing technical approach, analyses, and results (2000 ± 100 words). Self and peer evaluations along with specific contributions of each team member |

# Tasks

### Implementation: Use Python or Java. Python is recommended.

Phases of the project: Data Discovery, Extraction, Integration, Analytics, Validation, Reporting and Visualization.

## Data Discovery: 
Look for various types of data that are available about the City of Chicago, its people, businesses, health, transportation, land-use, government, etc from various sources. Each team should submit a list of at least ten sources along with a short description about the datasets provided and how it could be useful in Data Science. This list should exclude data.cityofchicago.org, datausa.io and census.gov. If a portal has multiple datasets, the portal should only be reported once with listing of all available datasets.

## Data Extraction: 
For the following datasets, we are interested in data for the zip codes from 60601 to 60607 (in Chicago):
- Business Licenses
- Crime
- Census
- Demographics from the City of Chicago’s data portal and census.gov.
- Scrape and use external data for restaurants from Yelp.
- **Optional (extra credit)**: Weather data.

## Data Integration and Analytics: 
Your framework should be able to address the following queries. Sample file formats for each of the following deliverables will be supplied in the next few weeks.

- Report of types of crime (Assault, Battery, etc) within 3 blocks of (a) Grocery Stores (b) Schools, and (c) Restaurants.
- Year, Business Type (a,b,c), Business Name, Address, Has Tobacco License, Has Liquor License, Crime Type, #Crimes, #Arrests, #On Premises

Using supervised learning, train a model to learn crime statistics and use it to predict the probability of different types of crime for a given set of addresses. Use three ML techniques (Logistic Regression, Decision Trees, and Random Forests) and compare their results.

**Attributes**:
Address, Crime Type, Technique, Probability

## Intersect parameters from multiple datasets: 
Create a graph that plots crime statistics for each age group by census block.
- Year, Census Block, Age Group, Crime Type, #Crimes

Determine the relationship (increase or decrease in ratings) between average review rating and food inspection result (pass/conditional pass/fail) for a restaurant.
- Restaurant Name, Address, Average Yelp Review, #Pass, #Conditional, #Failed Inspection

Analyze sentiments (positive, negative and neural) of each restaurant Yelp reviews
- Restaurant Name, Sentiment labels, Review Rating

Identify how the sentiments relate to the review rating by plotting average ratings against most frequent overall sentiments.
- Sentiment Term, Average Review Rating

Predict review rating using text from Yelp reviews. (Precision, recall and F1 score for competition)
- Review, Review Rating

What is the viability of a business, i.e., how long is a business active, after a failed food inspection?
- Restaurant Name, Address, Failed inspection on, Alive for x years

Does having a liquor license influence crime incidents in the neighborhood?
- Census Block, #Businesses with liquor licenses, #Crimes, #Arrests

**Optional (extra credit)**: Considering the past crime data and weather data for the given zip code, predict the probability of Robbery (include all types of robbery - aggravated/armed/…) for Summer 2018.
- Census Block, Month, Avg. Temperature, Type of Robbery (Aggravated/Armed/…), Probability

## Validation and Testing: 
- Each project team will manually build a reference dataset, using a small subset for each of the data integration and analytics tasks. - Each team may use their own dataset for internal testing, but may not share their datasets with other teams. The list of records from each team will be combined to create a gold standard. 
- The teaching team will also add their dataset to the gold standard. 
- Teams will be evaluated based on their framework’s performance compared to the gold standard. 
- We will use **precision, recall, and F-measure**, among other metrics. 
- For predictive analytics, we will use the **average of predicted result for each test record** as the gold standard, and each framework will be evaluated based on how much it differs from the gold standard.

## Visualization: 
- Build a website that displays the integrated data using a map such as OSM, ESRI or Google Maps. 
- This site should also include charts for all of the above data integration and analytics tasks. 
- Describe the datasets and ensure that there is an intuitive flow among the various components of the sites.
- Apply filters to show the results of the analytics performed. Examples: show multiple layers for each type of data (crime, population, etc) and allow the end-user to interact with data points.
- Charts and graphs should be **drawn dynamically based on various filter criteria**. You should not post a series of images that represent the reports.
- **Create a video explaining the data, the analysis your team performed, the capabilities of the application/framework you have developed, and show off the graphs and plot from your website.**

## In-class Presentation: 
- Each team will show their visualization to the class and describe its features. This allows everyone to see a variety of solutions to the problem, and a variety of implementations. 
- Rehearse your presentations several times.
- All team members are expected to participate equally in the presentation.


## Notes on Directory Structure
### Structure is as follows : 
/ will refer to the root of the repo, just as in Linux and contain the final few master scripts that will call scripts from /staging/*. Dont put anything there for now.

### The main directory in / is going to be /staging.

### /staging = the main directory where most of the scripts will be.
Its divided into lots of components for now, we can add or remove later as needed.

#### Inside /staging, there is a /staging/data where all the files will be stored. It contains 3 main folders :
- /staging/data/raw = store your temporary datasets here. These have to be cleaned, modified and standardized by scripts in /staging/* to get inside the clean /datasets
- /staging/data/datasets = stores only clean data which can directly be used by /staging
- /staging/data/models = stores the trained serialized models from /staging

#### Inside /staging root, only scripts which call upon imports from /staging/* will be kept in the root. 
These scripts will use the /staging/ml, /staging/preprocessing, /staging/utils to scrape websites, download data, store in /raw temporarily, clean data and store in /datasets, apply ml algos and store the trained models in /models

##### Now inside /staging/ml will be utils like cross validation, model training maybe, evaluation helpers (basically anything to do with ml)

##### Inside /staging/utils will be general utils needed by /staging, /staging/ml and /staging/preprocessing

##### Inside /staging/preprocessing we can put scripts that may be needed by /staging to clean up datasets from /raw and then save into /datasets

**Note** : **Always** use the method `resolve_data_path(path)` for read access to data inside /staging/data/*.
If you try direct access it will cause errors when we create the final scripts.

To use this function, directly use resolve_data_path('raw/{name}') or resolve_data_path('models/{name}') or resolve_data_path('datasets/{name}')

**Note** : **Always** use the method `construct_data_path(filename)` for write access to data inside /staging/data/*.
If you try direct access it will cause errors when we create the final scripts.

To use this function, directly use construct_data_path('raw/{name}') or resolve_data_path('models/{name}') or resolve_data_path('datasets/{name}')

