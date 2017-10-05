import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *
import movie_meta
import genome_scores


def basic_hist(column):
    np_column = column.as_matrix()
    np_column = np_column[~np.isnan(np_column)]
    plt.hist(np_column)
    plt.show()


def del_useless_tags(tags):
    del tags['tag']
    del tags['timestamp']
    del tags['userId']


def main():
    """
    mm = pd.read_csv('Data/movie_metadata.csv')
    mm = movie_meta.preprocess(mm)
    tags = pd.read_csv('Data/ratings_information/tags.csv')
    tags['tag_id'] = map_col_to_ind(
        pd.Series(data=tags['tag'].str.lower().astype(str).apply(lambda x: x.replace(' ', '')).astype(str), dtype=str))
    """
    gs = pd.read_csv('Data/genome-scores.csv')
    lk = pd.read_csv('Data/links.csv')
    gt = pd.read_csv('Data/genome-tags.csv')
    genome_scores.preprocess(gs)


if __name__ == '__main__':
    main()
