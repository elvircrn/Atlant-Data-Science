import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import movie_meta
import genome_scores
import links
import prototype
from sklearn.decomposition import PCA
from pandas import DataFrame
import ratings


def basic_hist(column):
    np_column = column.as_matrix()
    np_column = np_column[~np.isnan(np_column)]
    plt.hist(np_column)
    plt.show()


def merge_gtags(mov_meta, mov_gtags):
    return mov_gtags.merge(mov_meta, on='movieId')


def merge_link(mov_meta, link):
    return mov_meta.merge(link, on='imdbId')


def tuto():
    df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                             'foo', 'bar', 'foo', 'foo'],
                       'B': ['one', 'one', 'two', 'three',
                             'two', 'two', 'one', 'three'],
                       'C': np.random.randn(8),
                       'D': np.random.randn(8)})

    groups = df.groupby(by='A').size().reset_index(name='Size')
    print(groups)


def f():
    ratings = pd.read_csv('Data/ratings.csv')
    rating_groups = ratings.groupby(by='userId', sort=False).size()
    rating_groups = rating_groups.reset_index(name='Size')
    sorted_rating_groups = DataFrame.sort_values(rating_groups, by='Size', ascending=False)['Size']
    movie_count = np.array(sorted_rating_groups)

    print(sorted_rating_groups)
    # count, bins, ignored = plt.hist(movie_count, bins=1000)
    # plt.show()


def main():
    # mm = movie_meta.preprocess(pd.read_csv('Data/movie_metadata.csv'))
    # gs = genome_scores.preprocess(pd.read_csv('Data/genome-scores.csv'))
    # gt = pd.read_csv('Data/movie-gtags.csv')
    # lk = links.preprocess(pd.read_csv('Data/links.csv'))
    # mov_meta = merge_link(mm, lk)
    # mov_meta = merge_gtags(mov_meta, gt)
    # print('User cound: ', len(ratings['userId'].unique()))
    # print('Max movie id: ', lk['movieId'].max())
    # prototype.approx()
    r, y, mov_map, usr_map = ratings.get_rating_matrix()
    r = ratings.als(r, y)
    r.to_csv('Data/predictions.csv')
    print('Finished')


if __name__ == '__main__':
    main()
