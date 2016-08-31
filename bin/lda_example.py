#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import glob
import logging

import pandas as pd
logging.basicConfig(level=logging.INFO)

import easy_lda

def gen_movie_review_df():
    path_list = glob.glob('../data/input/pos/*.txt')
    review_df = pd.DataFrame(path_list, columns=['review_path'])
    review_df['review_text'] = review_df['review_path'].apply(lambda x: open(x, 'r').read())

    return review_df


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    review_df = gen_movie_review_df()

    X = review_df['review_text'].as_matrix()

    lda = easy_lda.LDA_topic_model(num_topics=5, num_iterations=500, clean_text=True, random_state=0)

    # Fit model
    lda.fit(X)

    # Get all data, in one convenient DataFrame
    print lda.transform_with_all_data_df(X)

    # Manually pull out probability of each document belonging to each topic
    print 'Topic probabilities for each document'
    print lda.predict_proba(X)

    # Manually pull out topic number
    print 'Topic number for each document'
    print lda.transform(X)

    # Manually pull out topic description
    print 'Topic descriptions'
    print lda.get_topic_descriptions()


# Main section
if __name__ == '__main__':
    main()
