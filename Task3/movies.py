import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *


def basic_hist(column):
    np_column = column.as_matrix()
    np_column = np_column[~np.isnan(np_column)]
    plt.hist(np_column)
    plt.show()


def main():

    movie_meta = pd.read_csv('Data/imdb_movie_information/movie_metadata.csv')
    tags = pd.read_csv('Data/ratings_information/tags.csv')
    print('Started parsing')
    tags['tag_id'] = map_col_to_ind(pd.Series(data=tags['tag'].str.lower().astype(str).apply(lambda x: x.replace(' ', '')).astype(str), dtype=str))

    del tags['tag']
    del tags['timestamp']
    del tags['userId']

    print(tags.groupby(by='movieId').count().sort_values(by='tag_id'))

    # print('Initial parsing done')
    # tag_id = map_col_to_ind(parsed_tag)
    # print(tag_id.value)

    # map_str_to_inds(data, ['color', 'director_name', 'actor_2_name', 'actor_3_name',

    # print(data['director_facebook_likes'])

    # basic_hist(data['gross'])
    # basic_hist(data['director_facebook_likes'])


if __name__ == '__main__':
    main()
