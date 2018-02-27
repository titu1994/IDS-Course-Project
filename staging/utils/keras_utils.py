import numpy as np
import os
import pickle
import logging
from typing import List, Set, Dict

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras import backend as K

from staging import resolve_data_path, construct_data_path
from staging.utils.text_utils import prepare_yelp_reviews_dataset_keras as _prepare_yelp_reviews_dataset_keras


EMBEDDING_DIM = 300
MAX_NB_WORDS = 105000
MAX_SEQUENCE_LENGTH = 280


'''
Below is a modification to the TensorBoard callback to perform 
batchwise writing to the tensorboard, instead of only at the end
of the batch.
'''
class TensorBoardBatch(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoardBatch, self).__init__(log_dir=log_dir,
                                               histogram_freq=histogram_freq,
                                               batch_size=batch_size,
                                               write_graph=write_graph,
                                               write_grads=write_grads,
                                               write_images=write_images,
                                               embeddings_freq=embeddings_freq,
                                               embeddings_layer_names=embeddings_layer_names,
                                               embeddings_metadata=embeddings_metadata)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, batch)

        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch * self.batch_size)

        self.writer.flush()


def fbeta_score(y_true, y_pred):
    '''
    Computes the fbeta score. For ease of use, beta is set to 1.
    Therefore always computes f1_score
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def load_embedding_matrix(word_index: Dict, max_nb_words: int, embedding_dim: int, print_error_words: bool=True) -> np.ndarray:
    '''
    Either loads the created embedding matrix at run time, or uses the
    GLoVe 840B word embedding to create a mini initialized embedding matrix
    for use by Keras Embedding layers

    Args:
        word_index: indices of all the words in the current corpus
        max_nb_words: maximum number of words in corpus
        embedding_dim: the size of the embedding dimension
        print_error_words: Optional, allows to print words from GLoVe
            that could not be parsed correctly.

    Returns:
        An Embedding matrix in numpy format
    '''
    path = 'models/embeddings/embedding_matrix.npy'
    path = construct_data_path(path)

    embedding_path = resolve_data_path('raw/embeddings/glove.840B.300d.txt')

    if not os.path.exists(path):
        embeddings_index = {}
        error_words = []

        logging.info('Creating embedding matrix..')
        logging.info('Glove Embeddings located at : %s' % embedding_path)

        # read the entire GLoVe embedding matrix
        f = open(embedding_path, encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except Exception:
                error_words.append(word)

        f.close()

        logging.info('Loaded %s word vectors.' % len(embeddings_index))

        # check for words that could not be loaded properly
        if len(error_words) > 0:
            logging.info("%d words could not be added because there was an issue during loading of vectors." % (len(error_words)))
            if print_error_words:
                logging.info("Words are : %s\n" % error_words)

        logging.info('Preparing embedding matrix.')

        word_added_count = 0
        # prepare embedding matrix
        nb_words = min(max_nb_words, len(word_index))
        embedding_matrix = np.zeros((nb_words, embedding_dim))
        for word, i in word_index.items():
            if i >= nb_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

                word_added_count += 1
                logging.info("Added word %s in embedding. Current count of words added = %d" % (word, word_added_count))

        logging.info('Number of words loaded in embedding matrix = %d. '
                     'Size of embedding matrix = %d' % (word_added_count, len(embedding_matrix)))

        # save the constructed embedding matrix in a file for efficient loading next time
        np.save(path, embedding_matrix)
        logging.info('Saved embedding matrix')

    else:
        # load pre-built embedding matrix
        embedding_matrix = np.load(path)
        logging.info('Loaded embedding matrix')

    return embedding_matrix


def load_prepared_embedding_matrix(finetuned: bool=False) -> np.ndarray:
    '''
    Loads a prepared embedding matrix. Will throw an exception if the file has not
    been created first !

    Returns:
        a prepared embedding matrix
    '''
    # load pre-built embedding matrix
    if not finetuned:
        embedding_name = 'embedding_matrix.npy'
    else:
        logging.info('Loading finetuned embedding matrix')
        embedding_name = 'finetuned_embedding_matrix.npy'

    path = 'models/embeddings/%s' % embedding_name
    path = resolve_data_path(path)

    embedding_matrix = np.load(path)
    logging.info('Loaded embedding matrix')

    return embedding_matrix


def create_ngram_set(input_list: List, ngram_value: int=2) -> Set:
    # construct n-gram text from uni-gram text input
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences: List, token_indice: Dict, ngram_range: int=2) -> List:
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def prepare_yelp_reviews_dataset_keras(path: str, max_nb_words: int, max_sequence_length: int,
                                       ngram_range: int=2, nb_sentiment_classes: int=3) -> (np.ndarray, np.ndarray, Dict):
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

    texts, labels = _prepare_yelp_reviews_dataset_keras(path, nb_sentiment_classes)

    labels = to_categorical(labels, num_classes=nb_sentiment_classes)

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

    return (data, labels, word_index)


if __name__ == '__main__':
    from staging import resolve_data_path
    logging.basicConfig(level=logging.INFO)

    reviews_path = resolve_data_path('raw/yelp-reviews/cleaned_yelp_reviews.csv')

    # construct the tokenizer
    data, labels, word_index = prepare_yelp_reviews_dataset_keras(reviews_path, max_nb_words=MAX_NB_WORDS,
                                                                  max_sequence_length=MAX_SEQUENCE_LENGTH,
                                                                  nb_sentiment_classes=3)

    # construct the actual embedding matrix
    embedding_matrix = load_embedding_matrix(word_index, MAX_NB_WORDS, EMBEDDING_DIM)
