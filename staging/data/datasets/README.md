# Contains cleaned dataset/s, ready for use in training or evaluating models

## Yelp Ratings

Data extracted by scraping https://www.yelp.com/search?find_loc=60601 and replicating 60601 to 60607 to scrape relevant information. <br>
Ideally will contain information about pincode, name of the restaurant, address, price (on average), ratings (in number of stars), reviews count and whether it serves alcohol or not.

This will allow us to identify the restaurants by their name, location and whether they serve alcohol or not.
This can be cross referenced with the larger Yelp dataset mentioned in the next dataset so that we can identify which restaurants are infact in Chicago, have a licence to sell alcohol, and augment the information required by the sentiment analysis classifier in later stages based on reviews.

### Attributes

| Attribute  | Description  |
| ------|:----------------:|
| id	| Unique ID for that row |
| Address  | Local chicago address, in string format with format [first address line, chicago, IL ***** <- pin code" |
| Category	| List of strings describing cuisine |
| Name	| Name of the restaurants |
| Neighbourhood	| String text describing the neighbourhood of the region (generally "The Loop") |
| Phone	| Phone number (in format (AREA CODE=312) (XXX)-(XXXX) |
| Ratings | Float rating ranging from 1.0 to 5.0, with intervals of 0.5 |
| Reviews | Number of reviews available |
| Alcohol | Crude test to see if they serve alcohol or not. Do not rely on it directly. Cross reference it with the Yelp Dataset ! |
| Pincode | Five digit integer pincode - ranging from 60601 to 60607 |

## Yelp Reviews

Data extracted by scraping https://www.yelp.com/biz/*, where the * was populated by names retrieved from the above mentioned "Yelp Ratings" dataset. <br>
This provides a clean textual dataset, which is small enough to quickly iterate and define the preprocessing and text processing stages for sentiment analysis.

It also provides a weak baseline on what we hope to improve using the full dataset obtained from the full Yelp Dataset below. <br>
Scraped nearly 250 of the 550 restaurants for approximately 58000 reviews, over the course of a day. <br>
Data is limited because Yelp has very aggressive throttling and IP bans for several minutes if caught scraping its pages.

### Attributes

| Attribute  | Description  |
| ------|:----------------:|
| id	| Unique ID for that row |
| rating  | Integer rating ranging from 1 to 5 for that review |
| restaurant_name	| String name of the restaurant |
| review	| Raw text review |

## Yelp Dataset

Full Yelp dataset of nearly 2.5 GB of data about 5.2 million reviews, 174000 businesses, 200000 images and accross 11 metropoliton areas (which includes Chicago). <br>

We will extract the relevant information the massive dataset, pertaining to our task which only focusses on Chicago. Doing so,
we can create a miniature dataset that will include the reviews of a large portion of the restaurants from pincode 60601 to 60607. Using this,
we can perform both sentiment analysis using the reviews, and also perhaps improve the performance system which checks if a restaurant is legally allowed to sell alcohol.

Since the dataset is so vast, and includes so many attributes, we will only list the major important ones, which will be used to link the distributed files by the appropriate ids.

### Attributes
These attributes and their descriptions are copied without modification from the official documentation of the Yelp dataset website. https://www.yelp.com/dataset/documentation/json

#### business.json

| Attribute  | Description  |
| ------|:----------------:|
| business_id	| 22 character unique business id |
| name  | the business's name |
| neighborhood	| the full address of the business (equivalent to Address in our "Yelp dataset") |
| postal code	| 5 digit integer postal code, ranging from 60601 to 60607 |
| stars	| Float rating ranging from 1.0 to 5.0, with intervals of 0.5 |
| review_count	| Number of reviews available |

### review.json

| Attribute  | Description  |
| ------|:----------------:|
| review_id	| 22 character unique review id |
| business_id	| 22 character business id, maps to business in business.json |
| text	| the review itself |

#### Remaining datasets

-   We do not use the user.json data in order to avoid implicit bias towards certain users.
-   checkins.json is not helpful for sentiment analysis
-   We are not interested in the number of tips etc
-   Photos of the food ect posted by the users for restaurant can implicitly bias the reviews. We will carefully consider whether their incorporation hurts or benefits the performance of the sentiment classifier.

We will only extract raw text from the rewiews themselves, without any external metadata for the task of sentiment analysis.