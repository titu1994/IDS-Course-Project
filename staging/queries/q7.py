import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "..")

import logging
logging.basicConfig(level=logging.INFO)

from staging import resolve_data_path
from staging.utils.sklearn_utils import compute_metrics
from staging.utils.sklearn_utils import RATINGS_CLASS_NAMES

if __name__ == '__main__':
    """
    NOTE: The models were trained on a 80-20 split of the original dataset.
    If you run them on the default dataset path as below, it will show extremely high 
    F1 scores. This is because the training set it included in the predictions.
    
    To get correct scores, first open q7_ml.py or q7_dl,
    edit the variable called "path" to the path of the golden dataset, and then run that script.
    
    It will generate a file which will be used below.
    
    NOTE 2:
    This evaluation above will of course need a trained model to be present in the correct
    directory of the "data" folder.
    
    Since the models are very large (especially for logistic regression), we can send them via drive.
    If you wish to train on your own, you will need to set up the proper data directory structure and
    the paths to your data files etc. 
    
    ****> We do not recommend it. <****
    It is simpler to just place the trained models in the correct subdirectory
    of the data directory.
    
    To train your own models, go to staging/ml_train/ and choose the logistic model training
    for ratings called : "train_yelp_ratings_logistic_regression.py". You will need to ensure that the
    input csv files have the correct column names etc.
    """

    # change this path if you changed the df path inside "evaluate_ml_restaurants_sentiment.py"
    # NOTE: Can change `dl` to `ml` below
    df_path = "results/yelp/dl_ratings_predictions.csv"
    df_path = resolve_data_path(df_path)

    df = pd.read_csv(df_path, encoding='latin-1', header=0)

    labels = df['Labels'].values
    predictions = df['Predictions'].values

    compute_metrics(labels, predictions, RATINGS_CLASS_NAMES)


