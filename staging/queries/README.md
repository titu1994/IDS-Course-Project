# NOTE I

- Use pip install -r "requirements.txt" at the root of the directory to install all of the required libraries.

- It is advisable to run all of these queries using PyCharm IDE or an equivalent modern IDE to load this project.
While it is possible to use this codebase from terminal or command line, it is not a straightforward method.

## For CMD / Terminal usage

Open terminal INSIDE the staging directory, but outside of all other subdirectories.
Then use the command as :

```python

python subdir/script_name.py

(eg) -> while inside the staging directory
python queries/q5.py
```

If you see the error referring to `staging could not be found`, it is because you are either inside the subdirectories where the scripts lie or are outside staging.

# NOTE II

For queries 4, 5, 6 and 7, it is required to create the directory structure as below:
`staging/data/results/yelp/` and place the 5 csv files inside that directory path.

# Q1

INPUT: python queries/q1.py [crimes.csv] [bl.csv]

where	crimes.csv = input crimes dataset
		bl.csv = input business licenses dataset

OUTPUT: q1-result.csv [approx 10MB]

All datasets are pre-filtered by ZIPCODE on data.cityofchicago.org.
ZIPCODE range: [60601-60607]

No columns need to be filtered, everything is filtered in code.

# Q2

INPUT: python queries/q1.py [crimes.csv]

where	crimes.csv = input crimes dataset

OUTPUT: q2-result.csv [approx 1.5GB]

All datasets are pre-filtered by ZIPCODE on data.cityofchicago.org.
ZIPCODE range: [60601-60607]

No columns need to be filtered, everything is filtered in code.

# Q4

INPUT : python queries/q4.py

With the file titles `inspection_ratings.csv` placed inside the above subdirectory, simply run the script.
No arguments are passed.

OUTPUT : The plot shows the average yelp reviews of the restaurants decreasing when they fail the inspection or pass with conditions.
The average rating fall is approximately 0.5 for PASS W/ CONDITIONS and approximately 1.5 for FAIL.

# Q5

INPUT : python queries/q5.py

NOTE: On Windows, command line terminal does not allow unicode characters on python 3.5 or below (this is fixed in 3.6).

> If you are on Windows on Python 3.5 or lower, please use the below command BEFORE launching the above python script in this order:

```
> set PYTHONIOENCODING=:replace

> python queries/q5.py
```

We do not accept command line arguments for this query, and instead choose to show the list of unique restaurants and then
ask the user at runtime to provide a restaurant name, so that the user can choose an appropriate restaurant from given selection.

OUTPUT : A Pandas Dataframe printout of the Restaurant Name, its relative position in the dataset, the sentiment labels associated
with that restaurant and the corresponding review rating.

# Q6

INPUT : python queries/q6.py

We do not accept command line arguments since it is a plotting script.

OUTPUT : Two diagrams -

1) A count plot of the number of instances of predicted ratings and predicted associated sentiment label. There is minor overlap
in the [3 - 4] rating range, since this range generally corresponds to "neutral" sentiment.

2) A facetgrid of the number of instances of predicted ratings and predicted associated sentiment labels. Here, the demarcation between
the range of [3 - 4] is more clearly visible in a side by side manner.

# Q7

INPUT : python queries/q7.py (direct inference)
INPUT : python queries/q7_ml.py OR python queries/q7_dl.py (prediction from pretrained models)

OUTPUT : For direct inference, you get fast f1 score metrics calculation along with a classification report

For prediction from pretrained models, they generate appropriate csv files that need to be parsed by `q7.py`
for retrieving the scores.

## NOTE

For prediction from pretrained models, you need the pretrained weights of the models, which for Logistic Regression (ML default) is
close to 6.5 GB in size. This is the reason this is not attached. If needed, please download the Logistic Model and the DL models from
the drive folder provided and place them in the following directories

> ML MODELS : 'data/models/sklearn/ratings/*.pkl' where * is the ml model name ;

> ML DEPENDENCIES : 'data/models/sklearn-utils/*.pkl' where * is the tfidf and vectorizer pickles

> DL MODELS : 'data/models/keras/ratings/weights/*.h5' where * is the dl model name ;

> DL MODELS : 'data/models/embeddings/*.npy' where * is the embedding matrix name

> DL MODELS : 'data/models/keras/sentiment/tokenizer.pkl' for the serialized Keras Tokenizer

For predictions, you need a csv file of the review text, which also contains the ratings.
The primary major columns that it should contain, with the exact same names are :

- review : The review text
- rating : the 1 - 5 rating scale in integer format

The filename inside the `path` variable at the top of `q7_ml.py` and `q7_dl.py` need to be changed to match the
directory structure and filename.

Foe ease of replication, the file should be placed in:
> 'data/datasets/yelp-reviews/reviews.csv'

If you only need to check our models, the Deep Learning models are much smaller, and faster to evaluate. However,
note that Logistic Regression is heavily positive biased and therefore it's scores will be higher in f1 ratings etc.

All of the deep learning models are trained to be unbiased estimators, and therefore while their raw f1 scores are lesser,
they are more intuitive in their understanding of the queries and predicting scores. Please keep this in mind when judging the models.


# Q10

INPUT: Replace line 16 with appropriate crime.csv filename and extension, line 34 with weather.csv

All datasets are pre-filtered by ZIPCODE on data.cityofchicag.org
ZIPCODE range: [60601-60607]

No columns need to be filtered, everything is filtered in code.

Census data is queried from CENSUS API.