import string
import os
import pickle
from typing import List
import logging

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-paper')
sns.set_style('white')

from staging import construct_data_path

# cache them to access faster
_vectorizer = None  # type: CountVectorizer
_tfidf_transformer = None  # type: TfidfTransformer


def load_dataset(path : str) -> pd.DataFrame:
    '''
    Loads the dataset as a pandas dataframe.
    Note: Must have a column called 'id' which is unique id for each row.

    Args:
        path: resolved path to the raw csv file

    Returns:
        pandas dataframe
    '''
    df = pd.read_csv(path, header=0, encoding='utf-8', index_col='id')
    return df


def add_sentiment_classes(df: pd.DataFrame, key: str, nb_classes: int = 3) -> pd.DataFrame:
    '''
    Adds the class label for sentiment analysis to
    Args:
        df: pandas dataframe
        key: column name to access the score
        nb_classes: number of classes for sentiment. Allowed = 2 or 3.

    Returns:
        dataframe augmented with 'class' column

    Raises:
        ValueErroe if nb_class is not one of 2 or 3

    '''
    if nb_classes not in [2, 3]:
        raise ValueError('`nb_classes` must be either binary (without neutral class) '
                         'or ternary (with neutral class)')

    if nb_classes == 3:
        def _class(rating):
            if rating > 3:
                return 2
            elif rating == 3:
                return 1
            else:
                return 0
    else:
        def _class(rating):
            if rating > 3:
                return 1
            else:
                return 0

    df['class'] = df[key].apply(_class)
    return df


def add_sentence_length(df: pd.DataFrame, key: str) -> pd.DataFrame:
    '''
    Adds the length of the sentence to the dataframe.

    Args:
        df: pandas dataframe
        key: column name to access the text column in the dataframe

    Returns:
        dataframe augmented with 'sentence_length' column
    '''
    df['sentence_length'] = df[key].apply(len)
    return df


def remove_stopwords_punctuation(df: pd.DataFrame, key: str, return_sentence: bool = True) -> pd.DataFrame:
    '''
    Performs simple cleaning tasks such as removing punctuation and stopwords.

    Args:
        df: pandas dataframe
        key: column name to access the text column in the dataframe
        return_sentence: whether to return a list of words or a reconstructed sentence

    Returns:
        dataframe with cleaned text in the key column
    '''
    # Ref: https://stackoverflow.com/questions/19130512/stopword-removal-with-nltk
    # Ref: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    def clean(text):
        text = clean_text(text)

        if return_sentence:
            text = ' '.join(text)

        return text

    df[key] = df[key].apply(clean)
    return df


def clean_text(text):
    '''
    Helper method to clean text

    Args:
        text: raw text

    Returns:
        cleaned text
    '''
    stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during',
                     'out',
                     'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such',
                     'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                     'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through',
                     'don',
                     'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while',
                     'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before',
                     'them',
                     'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because',
                     'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has',
                     'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't',
                     'being',
                     'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}
    stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

    # remove punctuation
    chars = [char for char in text
             if char not in string.punctuation]
    text = ''.join(chars)

    # remove stopwords
    text = [word for word in text.split()
            if word.lower() not in stopwords]

    return text


def tokenize(texts : List[str]) -> np.ndarray:
    '''
    For SKLearn models / XGBoost / Ensemble, use CountVectorizer to generate
    n-gram vectorized texts efficiently.

    Args:
        texts: input text sentences list

    Returns:
        the n-gram text
    '''
    global _vectorizer

    path = construct_data_path('models/sklearn-utils/vectorizer.pkl')

    if _vectorizer is None:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                logging.info('Vectorizer loaded from saved state !')
                vectorizer = pickle.load(f)
                _vectorizer = vectorizer
        else:
            logging.info('Building the Vectorizer..')

            vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=None)
            vectorizer.fit(texts)
            _vectorizer = vectorizer

            logging.info('Count Vectorizer finished building ! Saving to file now ..')

            with open(path, 'wb') as f:
                pickle.dump(vectorizer, f)
                logging.info('Count Vectorizer saved to file !')

        x_counts = _vectorizer.transform(texts)
    else:
        x_counts = _vectorizer.transform(texts)

    return x_counts


def tfidf(tokens) -> np.ndarray:
    '''
    Perform TF-IDF transform to normalize the dataset

    Args:
        x_counts: the n-gram tokenized sentences

    Returns:
        the TF-IDF transformed dataset
    '''
    global _tfidf_transformer

    path = construct_data_path('models/sklearn-utils/tfidf.pkl')

    if _tfidf_transformer is None:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                logging.info('TF-IDF transformer loaded from saved state !')
                transformer = pickle.load(f)
                _tfidf_transformer = transformer
        else:
            logging.info('Building the TF-IDF transformer..')

            transformer = TfidfTransformer()
            transformer.fit(tokens)
            _tfidf_transformer = transformer

            logging.info('TF-IDF transformer built. Saving to file now..')

            with open(path, 'wb') as f:
                pickle.dump(transformer, f)
                logging.info('TF-IDF transformer saved to file !')

        x_tfidf = _tfidf_transformer.transform(tokens)
    else:
        x_tfidf = _tfidf_transformer.transform(tokens)

    logging.info('Shape of tf-idf transformed datased : %s' % str(x_tfidf.shape))
    return x_tfidf


def prepare_yelp_reviews_dataset_sklearn(path : str, nb_sentiment_classes : int = 3) -> (np.array, np.array):
    '''
    Loads the yelp reviews dataset for Scikit-learn models,
    prepares them by adding the class label and cleaning the
    reviews, tokenize and normalize them.

    Args:
        path: resolved path to the dataset
        nb_sentiment_classes: number of sentiment classes.
            Can be 2 or 3 only.

    Returns:
        a tuple of (data, labels)
    '''
    df = load_dataset(path)

    # calculate the classes of the dataset
    df = add_sentiment_classes(df, key='rating', nb_classes=nb_sentiment_classes)

    # clean the dataset of stopwords and punctuations
    df = remove_stopwords_punctuation(df, key='review', return_sentence=True)

    # extract the cleaned reviews and the classes
    reviews = df['review']
    labels = df['class'].values

    tokens = tokenize(reviews)
    data = tfidf(tokens)

    return data, labels


def _prepare_yelp_reviews_dataset_keras(path : str, nb_sentiment_classes : int = 3) -> (np.array, np.array):
    '''
    Loads the yelp reviews dataset for Scikit-learn models,
    prepares them by adding the class label and cleaning the
    reviews, tokenize and normalize them.

    Args:
        path: resolved path to the dataset
        nb_sentiment_classes: number of sentiment classes.
            Can be 2 or 3 only.

    Returns:
        a tuple of (data, labels)
    '''
    df = load_dataset(path)

    # calculate the classes of the dataset
    df = add_sentiment_classes(df, key='rating', nb_classes=nb_sentiment_classes)

    # clean the dataset of stopwords and punctuations
    df = remove_stopwords_punctuation(df, key='review', return_sentence=True)

    # extract the cleaned reviews and the classes
    reviews = df['review'].values
    labels = df['class'].values

    return reviews, labels


def plot_yelp_dataset_info(df: pd.DataFrame):
    # Plot relationship of ratings and sentence length
    grid = sns.FacetGrid(data=df, col='rating')
    grid.map(plt.hist, 'sentence_length', bins=100)
    plt.show()

    # Observe the quantity of outliers when dealing with ratings and sequence lengths
    boxplot = sns.boxplot(x='rating', y='sentence_length', data=df, palette=sns.color_palette('GnBu_d'))
    plt.show()

    # plot relationship of class and sequence length
    grid = sns.FacetGrid(data=df, col='class')
    grid.map(plt.hist, 'sentence_length', bins=100)
    plt.show()

    # Observe the quantity of outliers when dealing with class and sequence lengths
    boxplot = sns.boxplot(x='class', y='sentence_length', data=df, palette=sns.color_palette('GnBu_d'))
    plt.show()


if __name__ == '__main__':
    from staging import resolve_data_path

    reviews_path = resolve_data_path('raw/yelp-reviews/cleaned_yelp_reviews.csv')
    #prepare_yelp_reviews_dataset_sklearn(reviews_path, nb_sentiment_classes=3)

    df = pd.read_csv(reviews_path, header=0, encoding='utf-8', index_col='id')
    df = add_sentence_length(df, key='review')
    df = add_sentiment_classes(df, key='rating', nb_classes=3)
    df = remove_stopwords_punctuation(df, key='review', return_sentence=True)

    print(df.info())
    print(df.describe())
    print(df.head())
    print()

    # names = df.restaurant_name.values
    # names, counts = np.unique(names, return_counts=True)

    # import csv
    # with open('restaurant_reviews.csv', 'w', encoding='utf-8', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['name', 'count'])
    #
    #     for name, count in zip(names, counts):
    #         writer.writerow([name, count])

    # for i, name in enumerate(names):
    #     print("Restaurant Name :", name)

    #plot_yelp_dataset_info(df)

