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


def get_mov_meta():
    mov_meta = movie_meta.preprocess(pd.read_csv('Data/movie_metadata.csv'))
    gs = genome_scores.preprocess(pd.read_csv('Data/genome-scores.csv'))
    gt = pd.read_csv('Data/movie-gtags.csv')
    lk = links.preprocess(pd.read_csv('Data/links.csv'))
    mov_meta = merge_link(mov_meta, lk)
    mov_meta = merge_gtags(mov_meta, gt)
    return movie_meta.del_duplicates(mov_meta)


def get_mov_meta_from_ratings(mov_meta, ratings):
    movie_ids = set(ratings['movieId'])
    mov_meta = mov_meta[mov_meta['movieId'].isin(movie_ids)]
    return mov_meta


def z_standard(mov_meta):
    mov_meta = mov_meta.fillna(0)
    std_scale = StandardScaler().fit(mov_meta)
    mov_meta = std_scale.transform(mov_meta)
    return mov_meta


def main():
    mov_meta = get_mov_meta()
    r, y, mov_map, usr_map, r_test, y_test, training_ratings = \
        ratings.get_rating_matrix(test_pct=0.20, mov_ids=set(mov_meta['movieId']))
    mov_meta = get_mov_meta_from_ratings(mov_meta, training_ratings)
    if len(training_ratings['movieId'].unique()) != len(mov_meta['movieId']):
        print('Id cnts don\'t match!{}x{}'.format(len(training_ratings['movieId'].unique()), len(mov_meta['movieId'])))
        return

    mov_meta = movie_meta.del_mov_ids(mov_meta)
    print('mov_meta -> {} x {}'.format(*mov_meta.shape))
    print('r -> {} x {}'.format(*r.shape))
    mov_meta = z_standard(mov_meta)


    X, Y, B = ratings.biased_als(r, y, R_test=r_test, W_test=y_test)
    # X, Y, B = ratings.als(r, y, R_test=r_test, W_test=y_test)



    # X, Y, B = ratings.als(r, y, R_test=r_test, W_test=y_test, Y=mov_meta.T)
    # pd.DataFrame(X).to_csv('Predictions/X.csv')
    # pd.DataFrame(Y).to_csv('Predictions/Y.csv')
    # pd.DataFrame(B).to_csv('Predictions/B.csv')
    print('Finished')


if __name__ == '__main__':
    main()
