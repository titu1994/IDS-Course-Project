import os
import numpy as np
import pandas as pd
import pickle
from typing import Dict

import logging
logging.basicConfig(level=logging.INFO)

from staging import resolve_data_path, construct_data_path

from staging.dl_eval.predict_dl_sentiment import get_lstm_sentiment_prediction
from staging.dl_eval.predict_dl_sentiment import get_multiplicative_lstm_sentiment_prediction
from staging.dl_eval.predict_dl_sentiment import get_malstm_fcn_sentiment_prediction
from staging.dl_eval.predict_dl_ratings import get_lstm_ratings_prediction
from staging.dl_eval.predict_dl_ratings import get_multiplicative_lstm_ratings_prediction
from staging.dl_eval.predict_dl_ratings import get_malstm_fcn_ratings_prediction

from staging.utils.text_utils import load_dataset, add_sentiment_classes, remove_stopwords_punctuation

from staging.utils.sklearn_utils import compute_metrics
from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES, RATINGS_CLASS_NAMES

from staging.utils.keras_utils import Tokenizer, to_categorical, pad_sequences, create_ngram_set, add_ngram
from staging.utils.keras_utils import MAX_SEQUENCE_LENGTH, MAX_NB_WORDS


def prepare_yelp_dataset_keras(path: str, max_nb_words: int, max_sequence_length: int,
                               ngram_range: int=2) -> (np.ndarray, np.ndarray, np.ndarray, Dict):
    '''
    Tokenize the data from sentences to list of words

    Args:
        path: resolved path to the dataset
        max_nb_words: maximum vocabulary size in text corpus
        max_sequence_length: maximum length of sentence
        ngram_range: n-gram of sentences
        nb_sentiment_classes: number of sentiment classes.
            Can be 2 or 3 only.

    Returns:
        A list of tokenized sentences and the word index list which
        maps words to an integer index.
    '''

    df = load_dataset(path)

    # calculate the classes of the dataset
    df = add_sentiment_classes(df, key='rating', nb_classes=2)

    # clean the dataset of stopwords and punctuations
    df = remove_stopwords_punctuation(df, key='review', return_sentence=True)

    # extract the cleaned reviews and the classes
    texts = df['review'].values
    labels = df['class'].values
    ratings = df['rating'].values
    restaurant_ids = df['restaurantID']

    labels = to_categorical(labels, num_classes=5)

    tokenizer_path = 'models/keras/sentiment/tokenizer.pkl'
    tokenizer_path = construct_data_path(tokenizer_path)

    if not os.path.exists(tokenizer_path): # check if a prepared tokenizer is available
        tokenizer = Tokenizer(num_words=max_nb_words)  # if not, create a new Tokenizer
        tokenizer.fit_on_texts(texts)  # prepare the word index map

        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)  # save the prepared tokenizer for fast access next time

        logging.info('Saved tokenizer.pkl')
    else:
        with open(tokenizer_path, 'rb') as f:  # simply load the prepared tokenizer
            tokenizer = pickle.load(f)
            logging.info('Loaded tokenizer.pkl')

    sequences = tokenizer.texts_to_sequences(texts)  # transform text into integer indices lists
    word_index = tokenizer.word_index  # obtain the word index map
    logging.info('Found %d unique 1-gram tokens.' % len(word_index))

    if ngram_range > 1:
        ngram_set = set()
        for input_list in sequences:
            for i in range(2, ngram_range + 1):  # prepare the n-gram sentences
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_nb_words + 1 if max_nb_words is not None else (len(word_index) + 1)
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}
        word_index.update(token_indice)

        max_features = np.max(list(indice_token.keys())) + 1  # compute maximum number of n-gram "words"
        logging.info('After N-gram augmentation, there are: %d features' % max_features)

        # Augmenting X_train and X_test with n-grams features
        sequences = add_ngram(sequences, token_indice, ngram_range)  # add n-gram features to original dataset

    logging.debug('Average sequence length: {}'.format(np.mean(list(map(len, sequences)), dtype=int))) # compute average sequence length
    logging.debug('Median sequence length: {}'.format(np.median(list(map(len, sequences))))) # compute median sequence length
    logging.debug('Max sequence length: {}'.format(np.max(list(map(len, sequences))))) # compute maximum sequence length

    data = pad_sequences(sequences, maxlen=max_sequence_length)  # pad the sequence to the user defined max length

    return (data, labels, ratings, restaurant_ids)



if __name__ == '__main__':
    # choose which predictor you would like ; default is logistic regression

    # predictor = get_lstm_sentiment_prediction
    predictor = get_multiplicative_lstm_sentiment_prediction
    # predictor = get_malstm_fcn_sentiment_prediction

    # Change the path here if needed by looking at the "data" directory
    path = "datasets/yelp-reviews/reviews.csv"
    path = resolve_data_path(path)

    # this dataset must match the contain two cols - "restaurantID" and "name"
    restaurant_data_path = "datasets/yelp-reviews/restaurants.csv"
    restaurant_data_path = resolve_data_path(restaurant_data_path)

    df = pd.read_csv(restaurant_data_path, header=0)
    df = df[['restaurantID', 'name']]

    print("Loading reviews dataset...")
    data, labels, ratings, ids = prepare_yelp_dataset_keras(path, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)

    labels = np.argmax(labels, axis=-1)

    min_rating = np.min(ratings)
    ratings = ratings - min_rating + 1

    print("Dataset prepared ! Calculating sentiment scores")
    predicted_reviews = predictor(data, False)[0]

    # predictor = get_lstm_ratings_prediction
    # predictor = get_multiplicative_lstm_ratings_prediction
    predictor = get_malstm_fcn_ratings_prediction

    print("Getting ratings scores")
    predicted_ratings = predictor(data, False)[0] + 1

    restaurant_names = []
    sentiment_prediction = []
    review_prediction = []

    for index in range(len(labels)):
        id = ids.iloc[index]  # get the restaurant id
        restaurant_name = df.loc[df['restaurantID'] == id, 'name'].values[0]  # get the restaurant name from the id

        restaurant_names.append(restaurant_name)
        sentiment_prediction.append(predicted_reviews[index])
        review_prediction.append(predicted_ratings[index])

    # build a dataframe and save it in a path
    df_path = "results/yelp/dl_sentiment_rating_query_result.csv"
    df_path = construct_data_path(df_path)

    dataframe = pd.DataFrame(data={
        'RestaurantName': restaurant_names,
        'SentimentLabels': sentiment_prediction,
        'ReviewRating': review_prediction
    }, columns=['RestaurantName', 'SentimentLabels', 'ReviewRating'])

    dataframe.to_csv(df_path, index=None)


    # compute the f1 scores of ratings here itself
    compute_metrics(labels, predicted_reviews, SENTIMENT_CLASS_NAMES)
    compute_metrics(ratings, predicted_ratings, RATINGS_CLASS_NAMES)





