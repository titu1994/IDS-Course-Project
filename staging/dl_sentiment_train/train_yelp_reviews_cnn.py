import numpy as np
import time

import sys
sys.path.insert(0, "..")

from staging import construct_data_path, resolve_data_path
from staging.utils.sklearn_utils import compute_metrics, create_train_test_set, compute_class_weight

from staging.utils.keras_utils import prepare_yelp_reviews_dataset_keras
from staging.utils.keras_utils import load_prepared_embedding_matrix
from staging.utils.keras_utils import fbeta_score, TensorBoardBatch

from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES, SENTIMENT_CLASS_PRIORS
from staging.utils.keras_utils import EMBEDDING_DIM, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH

from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation, Reshape
from keras.layers import Embedding, GlobalAveragePooling1D, concatenate, multiply
from keras.layers import Conv1D
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K


# edit the model name
MODEL_NAME = "cnn"
NB_EPOCHS = 20
BATCHSIZE = 512
REGULARIZATION_STRENGTH = 0.0

# constants that dont need to be changed
NB_SENTIMENT_CLASSES = 2
TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
LOG_STAMP = construct_data_path('models/keras/sentiment/logs/%s/%s' % (MODEL_NAME, TIMESTAMP))
WEIGHT_STAMP = construct_data_path('models/keras/sentiment/weights/%s_weights.h5' % (MODEL_NAME))

reviews_path = resolve_data_path('datasets/yelp-reviews/reviews.csv')
data, labels, _ = prepare_yelp_reviews_dataset_keras(reviews_path, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH,
                                                     nb_sentiment_classes=NB_SENTIMENT_CLASSES)

X_train, y_train, X_test, y_test = create_train_test_set(data, labels, test_size=0.1,
                                                         rebalance_class_distribution=False,
                                                         cache=False)

CLASS_WEIGHTS = 1. / np.asarray(SENTIMENT_CLASS_PRIORS)
print("Class weights : ", CLASS_WEIGHTS)

embedding_matrix = load_prepared_embedding_matrix(finetuned=False)
embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                            weights=[embedding_matrix],
                            trainable=False)


def squeeze_excite_block(input):
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


input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
x = embedding_layer(input)
x = Dropout(0.5)(x)

x = Conv1D(128, 8, padding='same', kernel_initializer='he_normal', dilation_rate=1,
           kernel_regularizer=l2(REGULARIZATION_STRENGTH))(x)
x = BatchNormalization(axis=-1)(x)
x = Activation('relu')(x)
x = squeeze_excite_block(x)

x = Conv1D(256, 5, padding='same', kernel_initializer='he_normal', dilation_rate=1,
           kernel_regularizer=l2(REGULARIZATION_STRENGTH))(x)
x = BatchNormalization(axis=-1)(x)
x = Activation('relu')(x)
x = squeeze_excite_block(x)

x = Conv1D(128, 3, padding='same', kernel_initializer='he_normal', dilation_rate=1,
           kernel_regularizer=l2(REGULARIZATION_STRENGTH))(x)
x = BatchNormalization(axis=-1)(x)
x = Activation('relu')(x)

x = GlobalAveragePooling1D()(x)
x = Dense(NB_SENTIMENT_CLASSES, activation='softmax', kernel_regularizer=l2(REGULARIZATION_STRENGTH))(x)

model = Model(input, x, name=MODEL_NAME)
model.summary()

optimizer = Adam(lr=5e-4, amsgrad=True)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy', fbeta_score])

# build the callbacks
checkpoint = ModelCheckpoint(WEIGHT_STAMP, monitor='val_fbeta_score', verbose=1, save_weights_only=True,
                             save_best_only=True, mode='max')
tensorboard = TensorBoardBatch(LOG_STAMP, batch_size=BATCHSIZE)
lr_scheduler = ReduceLROnPlateau(monitor='val_fbeta_score', factor=np.sqrt(0.5), patience=5, verbose=1,
                                 mode='min', min_lr=1e-5)

callbacks = [checkpoint, tensorboard, lr_scheduler]

# train model
model.fit(X_train, y_train, batch_size=BATCHSIZE, epochs=NB_EPOCHS, verbose=1,
          callbacks=callbacks, validation_data=(X_test, y_test), class_weight=CLASS_WEIGHTS)

# load up the best weights
model.load_weights(WEIGHT_STAMP)

scores = model.evaluate(X_test, y_test, batch_size=BATCHSIZE, verbose=1)

for (name, score) in zip(model.metrics_names, scores):
    print("%s -> %0.4f" % (name, score))

print()

predictions = model.predict(X_test, batch_size=BATCHSIZE, verbose=1)

y_test = np.argmax(y_test, axis=-1)
predictions = np.argmax(predictions, axis=-1)

compute_metrics(y_test, predictions, target_names=SENTIMENT_CLASS_NAMES)
