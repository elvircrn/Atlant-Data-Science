import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *
import re
from IPython.display import display, HTML


def basic_hist(column):
    np_column = column.as_matrix()
    np_column = np_column[~np.isnan(np_column)]
    plt.hist(np_column)
    plt.show()


def remove_imdb_prefix(link):
    result = re.search('[\\d]+', link)
    return result.group(0)


def main():
    movie_meta = pd.read_csv('Data/imdb_movie_information/movie_metadata.csv')
    tags = pd.read_csv('Data/ratings_information/tags.csv')
    print('Started parsing')
    tags['tag_id'] = map_col_to_ind(pd.Series(data=tags['tag'].str.lower().astype(str).apply(lambda x: x.replace(' ', '')).astype(str), dtype=str))

    del tags['tag']
    del tags['timestamp']
    del tags['userId']

    grouped_tags = tags.groupby(by='movieId')

    movie_meta['imdb_id'] = movie_meta['movie_imdb_link'].apply(lambda link: remove_imdb_prefix(link)).astype(str)
    del movie_meta['movie_imdb_link']

    # print(tags.groupby(by=['movieId', 'tag_id']).count().sort_values(by='tag_id').head(3))

    # print('Initial parsing done')
    # tag_id = map_col_to_ind(parsed_tag)
    # print(tag_id.value)

    keywords = set()

    for plot_keywords in movie_meta['plot_keywords'].astype(str).apply(lambda x: x.split('|')):
        for keyword in plot_keywords:
            keywords.add(keyword)

    print(len(keywords))
    print(movie_meta['title_year'].min())

    """
    
    data = map_str_to_inds(data, ['color',
                                  'director_name',
                                  'actor_2_name',
                                  'actor_3_name',
    
    """



    # print(data['director_facebook_likes'])

    # basic_hist(data['gross'])
    # basic_hist(data['director_facebook_likes'])


if __name__ == '__main__':
    main()
