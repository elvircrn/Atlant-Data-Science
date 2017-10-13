import numpy as np
import pandas as pd
from preprocessing import map_to_ind

from pandas import DataFrame


def avg(R):
    mu = 0
    cnt = 0
    for x in R:
        for y in x:
            if y > 0:
                cnt = cnt + 1
                mu += y
    return mu / cnt


def item_bias(R, D, U, mu):
    bq = np.zeros([D])  # Movie bias
    lambda1 = 25

    for i in range(D):
        Ri = 0
        s = 0
        for u in range(U):
            if R[u][i] > 0:
                Ri = Ri + 1
                s = s + R[u][i] - mu
        bq[i] = s / (lambda1 + Ri)

    return bq


def user_bias(R, D, U, mu, bq):
    bp = np.zeros([U])  # User bias
    lambda2 = 10
    for u in range(U):
        Ru = 0
        s = 0
        for i in range(D):
            if R[u][i] > 0:
                Ru = Ru + 1
                s = s + R[u][i] - mu - bq[i]
        bp[i] = s / (lambda2 + Ru)

    return bp


def get_bias(R, D, U):
    mu = avg(R)

    bq = item_bias(R, D, U, mu)
    bp = user_bias(R, D, U, mu, bq)  # User bias

    B = np.zeros([U, D])
    for i in range(0, U):
        for j in range(0, D):
            B[i][j] = mu + bq[j] + bp[i]
    return B


def get_error(R, W, X, Y):
    return np.sum((W * (R - np.dot(X, Y)) ** 2))


def get_reduced_ratings(ratings, count, by_movie=True):
    reduced_ratings = DataFrame()
    column = 'movieId' if by_movie else 'userId'
    entity_grps = ratings.groupby(by=column, sort=True).size().reset_index(name='Size')
    entity_grps = DataFrame.sort_values(entity_grps, by='Size', ascending=False)['Size']
    top_entities = DataFrame({column: pd.Series(entity_grps.index, index=None)})
    top_movies_set = set(top_entities[column].tolist()[1:count])
    reduced_ratings = ratings[ratings[column].isin(top_movies_set)]
    return reduced_ratings


def als(R, W, K=10, steps=3000):
    W = W.astype(np.float64, copy=False)
    R = R.astype(np.float64, copy=False)
    U = len(R)
    D = len(R[0])
    X = np.random.rand(U, K).astype(np.float64)
    Y = np.random.rand(K, D).astype(np.float64)
    B = get_bias(R, D, U).astype(np.float64)
    error_log = []
    _lambda = 0.001

    while steps > 0:
        # map(lambda u, Wu: np.linalg.solve(np.dot(Y, np.dot(Wu, Y.T)) + _lambda * np.eye(K),
        #                                     np.dot(Y, np.dot(Wu, R[u].T))).T, enumerate(X))
        for u in range(U):
            Wu = np.diag(W[u])
        X[u] = np.linalg.solve(np.dot(Y, np.dot(Wu, Y.T)) + _lambda * np.eye(K),
                               np.dot(Y, np.dot(Wu, R[u].T))).T
        for i in range(D):
            Wi = np.diag(W.T[i])
        Y[:, i] = np.linalg.solve(np.dot(X.T, np.dot(Wi, X)) + _lambda * np.eye(K),
                                  np.dot(X.T, np.dot(Wi, R[:, i])))
        err = get_error(R, W, X, Y)
        print(err)
        error_log.append(err)
        steps = steps - 1


def get_rating_matrix():
    ratings = pd.read_csv('Data/ratings.csv')
    ratings = get_reduced_ratings(ratings, 500)
    ratings = get_reduced_ratings(ratings, 500, by_movie=False)
    print(ratings['movieId'].max())
    print(ratings['userId'].max())
    num_movies = len(ratings['movieId'].unique())
    num_users = len(ratings['userId'].unique())
    mov_map = map_to_ind(ratings['movieId'].unique())
    user_map = map_to_ind(ratings['userId'].unique())
    r = np.zeros([num_users, num_movies])
    y = np.zeros([num_users, num_movies])
    for idx, rating in ratings.iterrows():
        i = user_map[int(rating['userId'])]
        j = mov_map[int(rating['movieId'])]
        y[i][j] = 1
        r[i][j] = rating['rating']
    return r, y, mov_map, user_map
