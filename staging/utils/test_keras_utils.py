import numpy as np
from keras import backend as K

from staging.utils import keras_utils


def test_integration_prepare_yelp_reviews():
    MAX_NB_WORDS = keras_utils.MAX_NB_WORDS
    MAX_SEQUENCE_LENGTH = keras_utils.MAX_SEQUENCE_LENGTH
    MAX_EMBEDDING_DIM = keras_utils.EMBEDDING_DIM

    reviews_path = keras_utils.resolve_data_path('raw/yelp-reviews/cleaned_yelp_reviews.csv')
    data, labels, word_index = keras_utils.prepare_yelp_reviews_dataset_keras(reviews_path, MAX_NB_WORDS,
                                                                              MAX_SEQUENCE_LENGTH,
                                                                              nb_sentiment_classes=3)
    embedding = keras_utils.load_embedding_matrix(word_index, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH,
                                                  print_error_words=False)

    assert data.shape[-1] == MAX_SEQUENCE_LENGTH
    assert labels.shape[-1] == 3
    assert embedding.shape == (MAX_NB_WORDS, MAX_EMBEDDING_DIM)


def test_load_pretrained_embedding_matrix():
    MAX_NB_WORDS = keras_utils.MAX_NB_WORDS
    MAX_EMBEDDING_DIM = keras_utils.EMBEDDING_DIM

    embedding = keras_utils.load_prepared_embedding_matrix(finetuned=False)
    assert embedding.shape == (MAX_NB_WORDS, MAX_EMBEDDING_DIM)

    embedding = keras_utils.load_prepared_embedding_matrix(finetuned=True)
    assert embedding.shape == (MAX_NB_WORDS, MAX_EMBEDDING_DIM)


def test_fbeta_score():
    y_true = np.zeros((10,))
    y_pred = np.zeros((10,))

    y_true[5:] = 1
    y_pred[2:7] = 1

    y_true = K.variable(y_true)
    y_pred = K.variable(y_pred)

    metric = keras_utils.fbeta_score(y_true, y_pred)
    f1_score = K.get_value(metric)

    assert metric is not None
    assert np.allclose([f1_score], [0.399999], rtol=1e-4)


def test_tensorboard():
    callback = keras_utils.TensorBoardBatch(histogram_freq=True,
                                            write_grads=True,
                                            write_graph=True,
                                            write_images=True,
                                            embeddings_freq=1)

    assert callback is not None