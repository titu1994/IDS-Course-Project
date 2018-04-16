import numpy as np
from typing import Union
import pickle

import sys
sys.path.insert(0, "..")

from staging import resolve_data_path
from staging.utils.text_utils import clean_text
from staging.utils.keras_utils import load_prepared_embedding_matrix

from staging.utils.keras_utils import MAX_SEQUENCE_LENGTH, MAX_NB_WORDS
from staging.utils.keras_utils import EMBEDDING_DIM

from staging.utils.layers import MultiplicativeLSTM, AttentionLSTM

from keras.layers import Dense, Input, BatchNormalization, Activation, Reshape, multiply
from keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, concatenate
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

# cache them to access faster multiple times
_tokenizer_ratings = None
_embedding_matrix_ratings = None
_lstm_model_ratings = None
_mlstm_model_ratings = None
_malstm_fcn_model_ratings = None

NB_RATINGS_CLASSES = 5


def _initialize():
    global _tokenizer_ratings, _embedding_matrix_ratings, _lstm_model_ratings, _mlstm_model_ratings, _malstm_fcn_model_ratings

    initialization_text = "default"
    initialization_text = _preprocess_text(initialization_text)  # will initialize the tokenizer

    if _embedding_matrix_ratings is None:
        _embedding_matrix_ratings = load_prepared_embedding_matrix(finetuned=False)


    if _lstm_model_ratings is None:
        embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                                    weights=[_embedding_matrix_ratings], trainable=False)

        input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        x = embedding_layer(input)
        x = Dropout(0.2)(x)
        x = LSTM(256)(x)
        x = Dense(NB_RATINGS_CLASSES, activation='softmax')(x)

        _lstm_model_ratings = Model(input, x, name="lstm_ratings")

        path = resolve_data_path('models/keras/ratings/weights/%s_weights.h5' % ('lstm'))
        _lstm_model_ratings.load_weights(path)

        _lstm_model_ratings.predict(initialization_text)

    if _mlstm_model_ratings is None:
        embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                                    weights=[_embedding_matrix_ratings], trainable=False)

        input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        x = embedding_layer(input)
        x = MultiplicativeLSTM(128)(x)
        x = Dense(NB_RATINGS_CLASSES, activation='softmax')(x)

        _mlstm_model_ratings = Model(input, x, name="lstm_ratings")

        path = resolve_data_path('models/keras/ratings/weights/%s_weights.h5' % ('mlstm'))
        _mlstm_model_ratings.load_weights(path)

        _mlstm_model_ratings.predict(initialization_text)


    if _malstm_fcn_model_ratings is None:
        embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                                    weights=[_embedding_matrix_ratings], trainable=False)

        input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embed = embedding_layer(input)
        x = Conv1D(100, 3, padding='same', kernel_initializer='he_uniform', strides=3)(embed)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = _squeeze_excite_block(x)

        x = AttentionLSTM(64)(x)

        y = Conv1D(96, 8, padding='same', kernel_initializer='he_uniform')(embed)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = _squeeze_excite_block(y)

        y = Conv1D(192, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = _squeeze_excite_block(y)

        y = Conv1D(96, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        x = Dense(NB_RATINGS_CLASSES, activation='softmax')(x)

        _malstm_fcn_model_ratings = Model(input, x, name="malstm_fcn_ratings")

        path = resolve_data_path('models/keras/ratings/weights/%s_weights.h5' % ('malstm_fcn'))
        _malstm_fcn_model_ratings.load_weights(path)

    print("Initialized deep learning models !")


def _preprocess_text(text: str):
    global _tokenizer_ratings

    text = clean_text(text)
    text = ' '.join(text)
    texts = [text]

    if _tokenizer_ratings is None:
        tokenizer_path = 'models/keras/sentiment/tokenizer.pkl'
        tokenizer_path = resolve_data_path(tokenizer_path)

        with open(tokenizer_path, 'rb') as f:  # simply load the prepared tokenizer
            _tokenizer_ratings = pickle.load(f)

    sequences = _tokenizer_ratings.texts_to_sequences(texts)  # transform text into integer indices lists
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # pad the sequence to the user defined max length

    return data


def _squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = K.int_shape(input)[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal')(se)
    se = multiply([input, se])
    return se


def get_lstm_ratings_prediction(text: Union[str, np.ndarray], preprocess: bool=True):
    global _embedding_matrix_ratings, _lstm_model_ratings

    if _embedding_matrix_ratings is None:
        _embedding_matrix_ratings = load_prepared_embedding_matrix(finetuned=False)

    if _lstm_model_ratings is None:
        embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                                    weights=[_embedding_matrix_ratings], trainable=False)

        input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        x = embedding_layer(input)
        x = Dropout(0.2)(x)
        x = LSTM(256)(x)
        x = Dense(NB_RATINGS_CLASSES, activation='softmax')(x)

        _lstm_model_ratings = Model(input, x, name="lstm_ratings")

        path = resolve_data_path('models/keras/ratings/weights/%s_weights.h5' % ('lstm'))
        _lstm_model_ratings.load_weights(path)

    if preprocess:
        data = _preprocess_text(text)
    else:
        data = text

    pred = _lstm_model_ratings.predict(data, batch_size=128)

    classification = np.argmax(pred, axis=-1)
    confidence = np.max(pred, axis=-1)

    return classification, confidence


def get_multiplicative_lstm_ratings_prediction(text: Union[str, np.ndarray], preprocess: bool=True):
    global _embedding_matrix_ratings, _mlstm_model_ratings

    if _embedding_matrix_ratings is None:
        _embedding_matrix_ratings = load_prepared_embedding_matrix(finetuned=False)

    if _mlstm_model_ratings is None:
        embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                                    weights=[_embedding_matrix_ratings], trainable=False)

        input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        x = embedding_layer(input)
        x = MultiplicativeLSTM(128)(x)
        x = Dense(NB_RATINGS_CLASSES, activation='softmax')(x)

        _mlstm_model_ratings = Model(input, x, name="lstm_ratings")

        path = resolve_data_path('models/keras/ratings/weights/%s_weights.h5' % ('mlstm'))
        _mlstm_model_ratings.load_weights(path)

    if preprocess:
        data = _preprocess_text(text)
    else:
        data = text

    pred = _mlstm_model_ratings.predict(data, batch_size=128)

    classification = np.argmax(pred, axis=-1)
    confidence = np.max(pred, axis=-1)

    return classification, confidence


def get_malstm_fcn_ratings_prediction(text: Union[str, np.ndarray], preprocess: bool=True):
    global _embedding_matrix_ratings, _malstm_fcn_model_ratings

    if _embedding_matrix_ratings is None:
        _embedding_matrix_ratings = load_prepared_embedding_matrix(finetuned=False)

    if _malstm_fcn_model_ratings is None:
        embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                                    weights=[_embedding_matrix_ratings], trainable=False)

        input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embed = embedding_layer(input)
        x = Conv1D(100, 3, padding='same', kernel_initializer='he_uniform', strides=3)(embed)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        x = _squeeze_excite_block(x)

        x = AttentionLSTM(64)(x)

        y = Conv1D(96, 8, padding='same', kernel_initializer='he_uniform')(embed)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = _squeeze_excite_block(y)

        y = Conv1D(192, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = _squeeze_excite_block(y)

        y = Conv1D(96, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        x = Dense(NB_RATINGS_CLASSES, activation='softmax')(x)

        _malstm_fcn_model_ratings = Model(input, x, name="malstm_fcn_ratings")

        path = resolve_data_path('models/keras/ratings/weights/%s_weights.h5' % ('malstm_fcn'))
        _malstm_fcn_model_ratings.load_weights(path)

    if preprocess:
        data = _preprocess_text(text)
    else:
        data = text

    pred = _malstm_fcn_model_ratings.predict(data, batch_size=128)

    classification = np.argmax(pred, axis=-1)
    confidence = np.max(pred, axis=-1)

    return classification, confidence


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    text = "This was very bad food !"
    label, confidence = get_malstm_fcn_ratings_prediction(text)

    print("Class = ", label, "Confidence:", confidence)

    text = "This was very good food ! Very happy "
    label, confidence = get_malstm_fcn_ratings_prediction(text)

    print("Class = ", label, "Confidence:", confidence)

    text = "What horrible food"
    label, confidence = get_malstm_fcn_ratings_prediction(text)

    print("Class = ", label, "Confidence:", confidence)