#!/usr/bin/env python
"""
coding=utf-8

Example code, showing how to utilize easy_lda.py to run topic modeling and predict topics.

"""
import glob
import logging

import pandas as pd

import easy_lda

logging.basicConfig(level=logging.INFO)


def main():
    """
    Main method of Topic Modeling example file, which shows appropriate use of easy_lda topic modeling, and
    useful functions
    :return: None
    :rtype: None
    """
    review_df = gen_movie_review_df()

    X = review_df['review_text'].as_matrix()

    topic_modeller = easy_lda.TopicModeler(num_topics=5, num_iterations=500, clean_text=True, random_state=0)

    # Fit model
    topic_modeller.fit(X)

    # Get all data, in one convenient DataFrame
    print topic_modeller.transform_with_all_data_df(X)

    # Manually pull out probability of each document belonging to each topic
    print 'Topic probabilities for each document'
    print topic_modeller.predict_proba(X)

    # Manually pull out topic number
    print 'Topic number for each document'
    print topic_modeller.transform(X)

    # Manually pull out topic description
    print 'Topic descriptions'
    print topic_modeller.get_topic_descriptions()


def gen_movie_review_df():
    """
    Generate the example data set, from the local file system. This data is courtesy
     https://www.cs.cornell.edu/people/pabo/movie-review-data/ , and contains the text from internet movie reviews.
    :return: Pandas DataFrame containing review text:
    :rtype: pd.DataFrame
    """
    path_list = glob.glob('../data/input/pos/*.txt')
    review_df = pd.DataFrame(path_list, columns=['review_path'])
    review_df['review_text'] = review_df['review_path'].apply(lambda x: open(x, 'r').read())

    return review_df

# Main section
if __name__ == '__main__':
    main()
