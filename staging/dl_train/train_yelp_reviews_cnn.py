import numpy as np
import time

from staging import construct_data_path, resolve_data_path
from staging.utils.sklearn_utils import compute_metrics, create_train_test_set, compute_class_weight

from staging.utils.keras_utils import prepare_yelp_reviews_dataset_keras
from staging.utils.keras_utils import load_prepared_embedding_matrix
from staging.utils.keras_utils import fbeta_score, TensorBoardBatch

from staging.utils.sklearn_utils import SENTIMENT_CLASS_NAMES
from staging.utils.keras_utils import EMBEDDING_DIM, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH

from keras.layers import Dense, Input, Dropout
from keras.layers import Embedding, GlobalAveragePooling1D
from keras.layers import Conv1D
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# edit the model name
MODEL_NAME = "cnn"
NB_EPOCHS = 5
BATCHSIZE = 128
REGULARIZATION_STRENGTH = 0.0001

# constants that dont need to be changed
NB_SENTIMENT_CLASSES = 3
TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
LOG_STAMP = construct_data_path('models/keras/sentiment/logs/%s/%s' % (MODEL_NAME, TIMESTAMP))
WEIGHT_STAMP = construct_data_path('models/keras/sentiment/weights/%s_weights.h5' % (MODEL_NAME))

reviews_path = resolve_data_path('raw/yelp-reviews/cleaned_yelp_reviews.csv')
data, labels, _ = prepare_yelp_reviews_dataset_keras(reviews_path, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH,
                                                     nb_sentiment_classes=NB_SENTIMENT_CLASSES)

X_train, y_train, X_test, y_test = create_train_test_set(data, labels, test_size=0.1)

#CLASS_WEIGHTS = compute_class_weight(np.argmax(y_train, axis=-1))
CLASS_WEIGHTS = [10., 1., 0.01]
print("Class weights : ", CLASS_WEIGHTS)

embedding_matrix = load_prepared_embedding_matrix()
embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, mask_zero=False,
                            weights=[embedding_matrix], trainable=False)

input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
x = embedding_layer(input)
x = Conv1D(128, 8, padding='same', activation='relu', kernel_initializer='he_normal',
           kernel_regularizer=l2(REGULARIZATION_STRENGTH))(x)
x = Conv1D(256, 5, padding='same', activation='relu', kernel_initializer='he_normal',
           kernel_regularizer=l2(REGULARIZATION_STRENGTH))(x)
x = Conv1D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal',
           kernel_regularizer=l2(REGULARIZATION_STRENGTH))(x)
x = GlobalAveragePooling1D()(x)
x = Dense(NB_SENTIMENT_CLASSES, activation='softmax', kernel_regularizer=l2(REGULARIZATION_STRENGTH))(x)

model = Model(input, x, name=MODEL_NAME)
model.summary()

optimizer = Adam(lr=1e-3, amsgrad=True)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy', fbeta_score])

# build the callbacks
checkpoint = ModelCheckpoint(WEIGHT_STAMP, monitor='val_fbeta_score', verbose=1, save_weights_only=True,
                             save_best_only=False, mode='max')
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
