import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *
import re
from functools import reduce


def basic_hist(column):
    np_column = column.as_matrix()
    np_column = np_column[~np.isnan(np_column)]
    plt.hist(np_column)
    plt.show()


def remove_imdb_prefix(link):
    result = re.search('[\\d]+', link)
    return result.group(0)


def binary_encode_tags(column):
    # .index()
    tag_list = list(get_tag_set(column))

    return column.apply(lambda x:
                        reduce(lambda a, b: a + b,
                               map(lambda tag: (1 << tag_list.index(tag)), x.split('|'))))


def del_useless_tags(tags):
    del tags['tag']
    del tags['timestamp']
    del tags['userId']


def del_useless_movie_meta(movie_meta):
    """
    NOTE: Always do this step last.
    """
    del movie_meta['movie_imdb_link']
    del movie_meta['movie_title']
    del movie_meta['aspect_ratio']


def map_strs(movie_meta):
    return map_str_to_inds(movie_meta, ['color',
                                        'director_name',
                                        'actor_1_name',
                                        'actor_2_name',
                                        'actor_3_name',
                                        # 'movie_title',
                                        'language',
                                        'country',
                                        'content_rating'])


def extract_imdb_id(column):
    return column.apply(lambda link: remove_imdb_prefix(link)).astype(str)


def movie_meta_fillna(movie_meta):
    """
    NOTE: Run this first
    """
    natozero = ['num_critic_for_reviews',
                'director_facebook_likes',
                'actor_1_facebook_likes',
                'actor_2_facebook_likes',
                'actor_3_facebook_likes',
                'gross',
                'movie_facebook_likes']
    for col_name in natozero:
        movie_meta[col_name] = movie_meta[col_name].fillna(0)

    duration_avg = movie_meta['duration'].mean()
    movie_meta['duration'] = movie_meta['duration'].fillna(duration_avg)

    year_avg = movie_meta['title_year'].mean()
    movie_meta['title_year'] = movie_meta['title_year'].fillna(year_avg)

    return movie_meta


def normalize_movie_meta(movie_meta):
    movie_meta['title_year'] = normalize(movie_meta['title_year'])
    movie_meta['budget'].apply(np.log10)
    return movie_meta


def main():
    movie_meta = pd.read_csv('Data/imdb_movie_information/movie_metadata.csv')
    tags = pd.read_csv('Data/ratings_information/tags.csv')
    print('Started parsing')
    tags['tag_id'] = map_col_to_ind(
        pd.Series(data=tags['tag'].str.lower().astype(str).apply(lambda x: x.replace(' ', '')).astype(str), dtype=str))
    print(binary_encode_tags(movie_meta['genres']))
    print(movie_meta['num_critic_for_reviews'])


if __name__ == '__main__':
    main()
