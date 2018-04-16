import numpy as np
from typing import Union
import pickle

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
_tokenizer_sentiment = None
_embedding_matrix_sentiment = None
_lstm_model_sentiment = None
_mlstm_model_sentiment = None
_malstm_fcn_model_sentiment = None

NB_SENTIMENT_CLASSES = 2


def _initialize():
    global _tokenizer_sentiment, _embedding_matrix_sentiment, _lstm_model_sentiment, _mlstm_model_sentiment, _malstm_fcn_model_sentiment

    initialization_text = "default"
    initialization_text = _preprocess_text(initialization_text)  # will initialize the tokenizer

    if _embedding_matrix_sentiment is None:
        _embedding_matrix_sentiment = load_prepared_embedding_matrix(finetuned=False)


    if _lstm_model_sentiment is None:
        embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                                    weights=[_embedding_matrix_sentiment], trainable=False)

        input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        x = embedding_layer(input)
        x = Dropout(0.2)(x)
        x = LSTM(256)(x)
        x = Dense(NB_SENTIMENT_CLASSES, activation='softmax')(x)

        _lstm_model_sentiment = Model(input, x, name="lstm_sentiment")

        path = resolve_data_path('models/keras/sentiment/weights/%s_weights.h5' % ('lstm'))
        _lstm_model_sentiment.load_weights(path)

        _lstm_model_sentiment.predict(initialization_text, batch_size=128)

    if _mlstm_model_sentiment is None:
        embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                                    weights=[_embedding_matrix_sentiment], trainable=False)

        input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        x = embedding_layer(input)
        x = MultiplicativeLSTM(128)(x)
        x = Dense(NB_SENTIMENT_CLASSES, activation='softmax')(x)

        _mlstm_model_sentiment = Model(input, x, name="lstm_sentiment")

        path = resolve_data_path('models/keras/sentiment/weights/%s_weights.h5' % ('mlstm'))
        _mlstm_model_sentiment.load_weights(path)

        _mlstm_model_sentiment.predict(initialization_text, batch_size=128)


    if _malstm_fcn_model_sentiment is None:
        embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                                    weights=[_embedding_matrix_sentiment], trainable=False)

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

        x = Dense(NB_SENTIMENT_CLASSES, activation='softmax')(x)

        _malstm_fcn_model_sentiment = Model(input, x, name="malstm_fcn_sentiment")

        path = resolve_data_path('models/keras/sentiment/weights/%s_weights.h5' % ('malstm_fcn'))
        _malstm_fcn_model_sentiment.load_weights(path)

    print("Initialized deep learning models !")


def _preprocess_text(text: str):
    global _tokenizer_sentiment

    text = clean_text(text)
    text = ' '.join(text)
    texts = [text]

    if _tokenizer_sentiment is None:
        tokenizer_path = 'models/keras/sentiment/tokenizer.pkl'
        tokenizer_path = resolve_data_path(tokenizer_path)

        with open(tokenizer_path, 'rb') as f:  # simply load the prepared tokenizer
            _tokenizer_sentiment = pickle.load(f)

    sequences = _tokenizer_sentiment.texts_to_sequences(texts)  # transform text into integer indices lists
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


def get_lstm_sentiment_prediction(text: Union[str, np.ndarray], preprocess: bool=True):
    global _embedding_matrix_sentiment, _lstm_model_sentiment

    if _embedding_matrix_sentiment is None:
        _embedding_matrix_sentiment = load_prepared_embedding_matrix(finetuned=False)

    if _lstm_model_sentiment is None:
        embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                                    weights=[_embedding_matrix_sentiment], trainable=False)

        input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        x = embedding_layer(input)
        x = Dropout(0.2)(x)
        x = LSTM(256)(x)
        x = Dense(NB_SENTIMENT_CLASSES, activation='softmax')(x)

        _lstm_model_sentiment = Model(input, x, name="lstm_sentiment")

        path = resolve_data_path('models/keras/sentiment/weights/%s_weights.h5' % ('lstm'))
        _lstm_model_sentiment.load_weights(path)

    if preprocess:
        data = _preprocess_text(text)
    else:
        data = text

    pred = _lstm_model_sentiment.predict(data)

    classification = np.argmax(pred, axis=-1)
    confidence = np.max(pred, axis=-1)

    return classification, confidence


def get_multiplicative_lstm_sentiment_prediction(text: Union[str, np.ndarray], preprocess: bool=True):
    global _embedding_matrix_sentiment, _mlstm_model_sentiment

    if _embedding_matrix_sentiment is None:
        _embedding_matrix_sentiment = load_prepared_embedding_matrix(finetuned=False)

    if _mlstm_model_sentiment is None:
        embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                                    weights=[_embedding_matrix_sentiment], trainable=False)

        input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        x = embedding_layer(input)
        x = MultiplicativeLSTM(128)(x)
        x = Dense(NB_SENTIMENT_CLASSES, activation='softmax')(x)

        _mlstm_model_sentiment = Model(input, x, name="lstm_sentiment")

        path = resolve_data_path('models/keras/sentiment/weights/%s_weights.h5' % ('mlstm'))
        _mlstm_model_sentiment.load_weights(path)

    if preprocess:
        data = _preprocess_text(text)
    else:
        data = text

    pred = _mlstm_model_sentiment.predict(data)

    classification = np.argmax(pred, axis=-1)
    confidence = np.max(pred, axis=-1)

    return classification, confidence


def get_malstm_fcn_sentiment_prediction(text: Union[str, np.ndarray], preprocess: bool=True):
    global _embedding_matrix_sentiment, _malstm_fcn_model_sentiment

    if _embedding_matrix_sentiment is None:
        _embedding_matrix_sentiment = load_prepared_embedding_matrix(finetuned=False)

    if _malstm_fcn_model_sentiment is None:
        embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                                    weights=[_embedding_matrix_sentiment], trainable=False)

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

        x = Dense(NB_SENTIMENT_CLASSES, activation='softmax')(x)

        _malstm_fcn_model_sentiment = Model(input, x, name="malstm_fcn_sentiment")

        path = resolve_data_path('models/keras/sentiment/weights/%s_weights.h5' % ('malstm_fcn'))
        _malstm_fcn_model_sentiment.load_weights(path)

    if preprocess:
        data = _preprocess_text(text)
    else:
        data = text

    pred = _malstm_fcn_model_sentiment.predict(data, batch_size=128)

    classification = np.argmax(pred, axis=-1)
    confidence = np.max(pred, axis=-1)

    return classification, confidence


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    text = "This was very bad food !"
    label, confidence = get_malstm_fcn_sentiment_prediction(text)

    print("Class = ", label, "Confidence:", confidence)

    text = "This was very good food !"
    label, confidence = get_malstm_fcn_sentiment_prediction(text)

    print("Class = ", label, "Confidence:", confidence)

    text = "What horrible food"
    label, confidence = get_malstm_fcn_sentiment_prediction(text)

    print("Class = ", label, "Confidence:", confidence)