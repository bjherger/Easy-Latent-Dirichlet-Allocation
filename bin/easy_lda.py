#!/usr/bin/env python
"""
coding=utf-8

Latent Dirichlet Allocation (LDA) topic modeling.

This code assists in generating topics from raw text, and assigning text to these topics.

"""
import logging
import re

import lda
import nltk
import numpy as np
import pandas as pd

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer


class TopicModeler(BaseEstimator):
    """
    An estimator to make topic modeling easier.

    Building on lda (http://pythonhosted.org/lda/), this estimator is designed to make it easier to:
     - clean text (if necessary)
     - train an LDA model
     - predict a topic for text

    """

    def __init__(self, num_topics=6, num_iterations=500, random_state=None, clean_text=True, vectorizer=None):
        """
        Init for LDA estimator
        :param num_topics: Number of topics to model (generally 3-10)
        :type num_topics: int
        :param num_iterations: Number of iterations to allow before locking in topics
        :type num_iterations: int
        :param random_state: Random seed, for consistent topics
        :type random_state: int
        :param clean_text: Whether to clean text using self.preprocess(). Recommended if you have not preprocessed
        the text already
        :type clean_text: bool
        :param vectorizer: Word vectorizer to use. The word vectorizer should convert a collection of text documents
        to a matrix of token counts
        """
        self.num_topics = num_topics
        self.num_iterations = num_iterations
        self.random_state = random_state
        self.lda_model = lda.LDA(n_topics=self.num_topics, n_iter=self.num_iterations, random_state=self.random_state)
        self.clean_text = clean_text
        self.get_topic_description_df = None
        if vectorizer is not None:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = CountVectorizer()

        # Make sure nltk has required data sets
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

    def fit(self, X):
        """
        Train the LDA model with the given documents. Each call to fit() will overwrite any previous calls to fit()
        :param X: List of documents, where each document is one string
        :type X: [str]
        :return: self
        :rtype: TopicModeler
        """
        logging.info('Fitting LDA Model')
        logging.debug('Input X: %s' % X)

        # Check if we should clean text
        if self.clean_text:
            logging.info('Cleaning text')
            X = map(self.preprocess, X)
            logging.debug('Cleaned text: %s' % X)

        # Vectorize data
        logging.info('Vectorizing data')
        vectorizer = self.vectorizer
        doc_matrix = vectorizer.fit_transform(X)
        logging.debug('Vectorized data: %s' % doc_matrix)

        # Create LDA model
        logging.info('Fitting LDA model')
        model = self.lda_model
        model.fit(doc_matrix)

        logging.info('Creating topic descriptions')
        self.set_topic_descriptions()
        logging.info('Topic descriptions: %s ' % self.get_topic_description_df)

        return self

    def transform(self, X, max_iter=20, tol=1e-16):
        """
        Returns the most likely topic for each document in X
        :param X: List of documents, where each document is one string
        :type X: [str]
        :param max_iter: The maximum number of iterations to allow for transformation. This should be monitored to check
        if log likelihood is decreasing.
        :type max_iter: int
        :param tol:  Tolerance value used in stopping condition.
        :type tol: float
        :return: List of assigned topic numbers, numbered beginning at index 0
        :rtype: [int]
        """

        if self.clean_text:
            logging.info('Cleaning text')
            X = map(self.preprocess, X)
            logging.debug('Cleaned text: %s' % X)

        # Get raw probabilities for each topic
        logging.info('Getting raw probabilities for each topic')
        topic_probabilities = self.predict_proba(X, max_iter, tol)

        # Find index of most likely topic
        logging.info('Finding index of most likely topic')
        topic_numbers = map(lambda x: x.argmax(), topic_probabilities)

        # Return results
        return topic_numbers

    def predict_proba(self, X, max_iter=20, tol=1e-16):
        """
        Predict the probability of the document being in each topic. Each row is one document, each observation is the
        probability of the document mapping the to the topic at the index. For example:
        [[.5, .2, .3],
         [.1, .7, .2]]

        This would suggest that the 0th document has a .5 probability of belonging to topic 0, .2 probability of
        belonging to topic 1, and .3 probability of belonging to topic 2. [.1, .7, .2] represents the probabilities for
         document 1.

        :param X: List of documents, where each document is one string
        :type X: [str]
        :param max_iter: The maximum number of iterations to allow for transformation. This should be monitored to check
        if log likelihood is decreasing.
        :type max_iter: int
        :param tol:  Tolerance value used in stopping condition.
        :type tol: float
        :return: Matrix containing probability for each document, for each topic
        """
        if self.clean_text:
            X = map(self.preprocess, X)

        # Vectorize data
        vectorizer = self.vectorizer
        doc_matrix = vectorizer.fit_transform(X)

        topics = self.lda_model.transform(doc_matrix, max_iter, tol)
        return topics

    def get_topic_descriptions(self, n_top_words=10):
        """
        Returns the top n words associated with each topic. Words are ordered by how distinct they are to the topic.

        Topic numbers are denoted by the index. For example:
        [['computer', 'science', 'technology'],
         ['farming', 'breeding', 'livestock']]

         The list ['computer', 'science', 'technology'] refers to the 0th topic, and the list
         ['farming', 'breeding', 'livestock'] refers to the first topic.

        :param n_top_words: Number of canonical words to return for each topic
        :type n_top_words: int
        :return: list of lists, containing canonical words
        :rtype: [[str]]
        """
        logging.info('Creating topic top words df')

        topic_words_df = self.get_topic_description_df
        logging.debug('Finished creating topic top words df: \n' + str(topic_words_df))

        # Convert to matrix
        # TODO return list instead of combined string
        return_matrix = topic_words_df['top_words'].tolist()

        # Return
        return return_matrix

    def set_topic_descriptions(self, n_top_words=10):
        """
        Set the topic description. The topic description is set during fit(), to avoid issues with changing
        vocabulary / vectorizer issues.
        :param n_top_words: Number of top words to include
        :return: None
        :rtype: None
        """
        # Convert vocabulary from {'word': index} to array with every word in its index (e.g. ['the', 'fat', 'man'])
        vocab_dict = self.vectorizer.vocabulary_
        vocab = [None] * (max(vocab_dict.values()) + 1)

        for key, value in vocab_dict.iteritems():
            vocab[value] = key

        # Pull topic_word object from lda model
        topic_word = self.lda_model.topic_word_

        # Create DataFrame containing top words for every topic
        topic_words_list = list()
        for topic_index, topic_word_values in enumerate(topic_word):
            ordered_vocab = np.array(vocab)[np.argsort(topic_word_values)]
            topic_words = ordered_vocab[:-n_top_words:-1]
            local_dict = dict()
            local_dict['topic_number'] = topic_index
            local_dict['top_words_list'] = topic_words

            topic_words_list.append(local_dict)

        topic_words_df = pd.DataFrame(topic_words_list)
        topic_words_df['top_words'] = topic_words_df['top_words_list'].apply(lambda x: ' '.join(x))

        self.get_topic_description_df = topic_words_df

    @staticmethod
    def preprocess(input_str):
        """
        Preprocess individual strings, including:
         - Normalize text (Lowercase, remove characters that are not letter or whitespace)
         - Tokenize (Break long string into individual document_words)
         - Lemmatize (Normalize similar document_words)
         - Remove custom stopwords (Remove document_words that have little value in this context)
         - Rejoin (Make everything back into one long string)
        :param input_str: raw string
        :type input_str: unicode
        :return: String, containing normalized text
        :rtype: unicode
        """

        logging.debug('Preprocessing string: ' + input_str)

        # Remove non-alphabetical / whitespace characters
        input_str = re.sub(r'[-.!]', ' ', input_str)
        input_str = re.sub(r'[^\s^\w]', '', input_str)
        input_str = re.sub(r'[0-9]', '', input_str)

        # Lowercase
        input_str = input_str.lower()

        # Tokenize (break up into individual document words)
        document_words = nltk.word_tokenize(input_str)

        # Lemmatize (e.g. friends -> friend, walking -> walk)
        lemmatizer = WordNetLemmatizer()
        stems = list()
        for word in document_words:
            stems.append(lemmatizer.lemmatize(word))

        # Get standard list of english stop words
        stop = stopwords.words('english')

        # Add any additional stopwords
        stop.extend(['the'])

        # Strip stopwords (they have little value in this context)
        stems = filter(lambda x: x not in stop, stems)

        # Reset to one long string, so that vectorizer won't complain
        output = u' '.join(stems)

        logging.debug('Done preprocessing string, result is: ' + output)

        return output

    def transform_with_all_data_df(self, X, max_iter=20, tol=1e-16, n_top_words=10):
        """
        Create a Pandas DataFrame containing the following:
         - input_text: Text, as it was when it was input
         - normalized_text: Text that was used for modeling. This may be the same as input_text, if clean_text = False
         - topic_number: The topic most associated with this document
         - topic_description: The words most associated with the topic associated with this document
         - topic_i: (for integer values of i) The probability that this document belongs to topic i
        :param X: List of documents, where each document is one string
        :type X: [str]
        :param max_iter: The maximum number of iterations to allow for transformation. This should be monitored to check
        if log likelihood is decreasing.
        :param tol:  Tolerance value used in stopping condition.
        :type tol: float
        :param n_top_words: Number of canonical words to return for each topic
        :type n_top_words: int
        :return: DataFrame containing fields described in docstring
        :rtype: pd.DataFrame
        """
        return_df = pd.DataFrame(X, columns=['input_text'])
        if self.clean_text:
            X = map(self.preprocess, X)
        return_df['normalized_text'] = X
        probability_col_names = map(lambda x: 'topic_' + str(x), range(0, self.num_topics, 1))
        topic_probabilities_df = pd.DataFrame(self.predict_proba(X, max_iter, tol), columns=probability_col_names)
        topic_number_df = pd.DataFrame(self.transform(X, max_iter, tol), columns=['topic_number'])

        # Join probabilities, topic number on index
        return_df = return_df.join(other=topic_probabilities_df)
        return_df = return_df.join(other=topic_number_df)

        topic_description_df = pd.DataFrame(self.get_topic_descriptions(n_top_words), columns=['topic_description'])
        topic_description_df['topic_number'] = topic_description_df.index.tolist()

        print topic_description_df
        print topic_description_df.describe(include='all')

        # Join on topic number
        return_df = pd.merge(left=return_df, right=topic_description_df, on='topic_number', how='left')

        return return_df
