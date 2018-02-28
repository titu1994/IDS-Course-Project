import numpy as np
from staging.utils import text_utils
from staging import resolve_data_path


def test_load_dataset():
    reviews_path = resolve_data_path('raw/yelp-reviews/cleaned_yelp_reviews.csv')
    df = text_utils.load_dataset(reviews_path)

    assert df.shape[-1] == 3


def test_add_sentiment_classes():
    reviews_path = resolve_data_path('raw/yelp-reviews/cleaned_yelp_reviews.csv')
    df = text_utils.load_dataset(reviews_path)

    df2 = text_utils.add_sentiment_classes(df, key='rating', nb_classes=2)
    df2_classes = np.unique(df2['class'].values)

    df3 = text_utils.add_sentiment_classes(df, key='rating', nb_classes=3)
    df3_classes = np.unique(df3['class'].values)

    assert len(df2_classes) == 2
    assert len(df3_classes) == 3


def test_add_sentence_length():
    reviews_path = resolve_data_path('raw/yelp-reviews/cleaned_yelp_reviews.csv')
    df = text_utils.load_dataset(reviews_path)

    df = text_utils.add_sentence_length(df, key='review')
    assert df.shape[-1] == 4


def test_prepare_yelp_reviews_dataset_sklearn():
    reviews_path = resolve_data_path('raw/yelp-reviews/cleaned_yelp_reviews.csv')

    data, labels = text_utils.prepare_yelp_reviews_dataset_sklearn(reviews_path, nb_sentiment_classes=3)

    classes = len(np.unique(labels))
    assert classes == 3

    data, labels = text_utils.prepare_yelp_reviews_dataset_sklearn(reviews_path, nb_sentiment_classes=2)

    classes = len(np.unique(labels))
    assert classes == 2


def test_prepare_yelp_reviews_dataset_keras():
    reviews_path = resolve_data_path('raw/yelp-reviews/cleaned_yelp_reviews.csv')

    data, labels = text_utils.prepare_yelp_reviews_dataset_keras(reviews_path, nb_sentiment_classes=3)

    classes = len(np.unique(labels))
    assert classes == 3

    data, labels = text_utils.prepare_yelp_reviews_dataset_keras(reviews_path, nb_sentiment_classes=2)

    classes = len(np.unique(labels))
    assert classes == 2
