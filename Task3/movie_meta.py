import numpy as np
from preprocessing import *
import re
from functools import reduce


def remove_imdb_prefix(link):
    result = re.search('[\\d]+', link)
    return result.group(0)


def binary_encode_movie_genres(movie_meta):
    tag_list = list(get_tag_set(movie_meta['genres']))

    movie_meta['genres'] = movie_meta['genres'] \
        .apply(lambda x:
               reduce(lambda a, b: a + b,
                      map(lambda tag: (1 << tag_list.index(tag)),
                          x.split('|')))).astype(int)
    return movie_meta


def del_useless_movie_meta(movie_meta):
    """
    NOTE: Always do this step last.
    """
    del movie_meta['movie_imdb_link']
    del movie_meta['movie_title']
    del movie_meta['aspect_ratio']
    del movie_meta['plot_keywords']
    return movie_meta


def map_mov_meta_strs_to_ind(movie_meta):
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
    return column.apply(lambda link: remove_imdb_prefix(link)).astype('str')


def movie_meta_fillna(movie_meta):
    """
    NOTE: Run this first
    """
    natozero = ['num_critic_for_reviews',
                'num_user_for_reviews',
                'director_facebook_likes',
                'actor_1_facebook_likes',
                'actor_2_facebook_likes',
                'actor_3_facebook_likes',
                'gross',
                'movie_facebook_likes',
                'budget']
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


def movie_str_preprocess(movie_meta):
    movie_meta['imdbId'] = extract_imdb_id(movie_meta['movie_imdb_link'])
    return movie_meta


def get_plot(plot_keywords, tag_dict, index):
    return plot_keywords.astype(str).apply(
        lambda tags: 0 if len(tags.split('|')) <= index else tag_dict[tags.split('|')[index]])


def map_plot_keywords(movie_meta):
    tag_dict = get_tag_dict(movie_meta['plot_keywords'])
    for i in range(5):
        movie_meta['plot_keyword' + str(i)] = get_plot(movie_meta['plot_keywords'], tag_dict, i)
    return movie_meta


def preprocess(movie_meta):
    return (del_useless_movie_meta
            (map_plot_keywords
             (binary_encode_movie_genres
              (movie_str_preprocess
               (normalize_movie_meta
                (map_mov_meta_strs_to_ind
                 (movie_meta_fillna(movie_meta))))))))
