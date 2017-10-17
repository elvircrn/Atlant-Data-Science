import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import movie_meta
import genome_scores
import links
import prototype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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


def get_mov_meta():
    mm = movie_meta.preprocess(pd.read_csv('Data/movie_metadata.csv'))
    gs = genome_scores.preprocess(pd.read_csv('Data/genome-scores.csv'))
    gt = pd.read_csv('Data/movie-gtags.csv')
    lk = links.preprocess(pd.read_csv('Data/links.csv'))
    mov_meta = merge_link(mm, lk)
    mov_meta = merge_gtags(mov_meta, gt)
    return mov_meta


def get_mov_meta_for_ratings(mov_meta, ratings):
    return mov_meta


def main():
    mov_meta = get_mov_meta()
    prototype.approx()

    r, y, mov_map, usr_map, r_test, y_test = ratings.get_rating_matrix(test_pct=0.20)
    print(np.sum(y_test))

    mov_meta = get_mov_meta_for_ratings(mov_meta, ratings)
    mov_meta = mov_meta.fillna(0)
    std_scale = StandardScaler().fit(mov_meta)
    mov_meta = std_scale.transform(mov_meta)
    print('mov_meta -> {} x {}'.format(len(mov_meta), len(mov_meta[0]))) # K -> 37

    X, Y, B = ratings.als(r, y, R_test=r_test, W_test=y_test, Y=mov_meta.T)
    # pd.DataFrame(X).to_csv('Predictions/X.csv')
    # pd.DataFrame(Y).to_csv('Predictions/Y.csv')
    # pd.DataFrame(B).to_csv('Predictions/B.csv')
    print('Finished')


if __name__ == '__main__':
    main()
