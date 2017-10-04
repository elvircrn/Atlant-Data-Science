import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *
import movie_meta


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
    mm = pd.read_csv('Data/imdb_movie_information/movie_metadata.csv')
    mm = movie_meta.preprocess(mm)
    tags = pd.read_csv('Data/ratings_information/tags.csv')
    tags['tag_id'] = map_col_to_ind(
        pd.Series(data=tags['tag'].str.lower().astype(str).apply(lambda x: x.replace(' ', '')).astype(str), dtype=str))


if __name__ == '__main__':
    main()
