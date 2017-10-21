import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        bp[u] = s / (lambda2 + Ru)

    return bp


def get_bias(R, D, U):
    mu = avg(R)

    bq = item_bias(R, D, U, mu)
    bp = user_bias(R, D, U, mu, bq)

    B = np.zeros([U, D])
    for i in range(0, U):
        for j in range(0, D):
            B[i][j] = mu + bq[j] + bp[i]
    return B


def get_error(R, W, X, Y):
    return np.sum((W * (R - np.dot(X, Y)) ** 2))


def get_biased_error(R, W, X, Y, B):
    return np.sum((W * (R - B - np.dot(X, Y)) ** 2))


def reduce_ratings(ratings, count, by_movie=True):
    column = 'movieId' if by_movie else 'userId'
    entity_grps = ratings.groupby(by=column, sort=True).size().reset_index(name='Size')
    entity_grps = DataFrame.sort_values(entity_grps, by='Size', ascending=False)['Size']
    top_entities = DataFrame({column: pd.Series(entity_grps.index, index=None)})
    top_movies_set = set(top_entities[column].tolist()[1:count])
    reduced_ratings = ratings[ratings[column].isin(top_movies_set)]
    return reduced_ratings


def biased_als(R, W, K=120, steps=3000, R_test=None, W_test=None, Y=None, biased=False):
    if (R_test is None) ^ (W_test is None):
        raise ValueError('R_test and W_test have to be either None or not None')
    elif R_test is not None:
        W_test = W_test.astype(np.float64, copy=False)
        R_test = R_test.astype(np.float64, copy=False)

    fix_movies = False
    U, D = R.shape

    if Y is not None:
        fix_movies = True
        K, _ = Y.shape
    else:
        Y = 5 * np.random.rand(K, D).astype(np.float64, copy=False)

    W = W.astype(np.float64, copy=False)
    R = R.astype(np.float64, copy=False)
    X = 5 * np.random.rand(U, K).astype(np.float64, copy=False)
    B = get_bias(R, D, U).astype(np.float64, copy=False)
    error_log = []
    error_test_log = []
    _lambda = 0.05

    if biased:
        R = R - B

    if fix_movies:
        for u in range(U):
            Wu = np.diag(W[u])
            X[u] = np.linalg.solve(np.dot(Y, np.dot(Wu, Y.T)) + _lambda * np.eye(K),
                                   np.dot(Y, np.dot(Wu, R[u].T))).T
            err = get_biased_error(R, W, X, Y, B) if biased else get_error(R, W, X, Y)

    err = np.inf
    while steps > 0 and err > 0.002:
        for u in range(U):
            Wu = np.diag(W[u])
            X[u] = np.linalg.solve(np.dot(Y, np.dot(Wu, Y.T)) + _lambda * np.eye(K),
                                   np.dot(Y, np.dot(Wu, R[u].T))).T
        if not fix_movies:
            for i in range(D):
                Wi = np.diag(W.T[i])
                Y[:, i] = np.linalg.solve(np.dot(X.T, np.dot(Wi, X)) + _lambda * np.eye(K),
                                          np.dot(X.T, np.dot(Wi, R[:, i])))

        err = get_biased_error(R, W, X, Y, B) if biased else get_error(R, W, X, Y)
        error_log.append(err)
        print('Error: {}'.format(err))

        if R_test is not None:
            err_test = get_biased_error(R_test, W_test, X, Y, B) if biased else get_error(R_test, W_test, X, Y)
            error_test_log.append(err_test)
            print('Test Error: {}'.format(err_test))

        steps = steps - 1

    plt.plot(error_log)
    if R_test is not None:
        plt.plot(error_test_log, 'r')
    plt.title('Learning RMSE')
    plt.xlabel('Iteration count')
    plt.ylabel('Error')
    plt.show()

    return X, Y, B


def als(R, W, K=120, steps=3000, R_test=None, W_test=None, Y=None, biased=False):
    if (R_test is None) ^ (W_test is None):
        raise ValueError('R_test and W_test have to be either None or not None')
    elif R_test is not None:
        W_test = W_test.astype(np.float64, copy=False)
        R_test = R_test.astype(np.float64, copy=False)

    fix_movies = False
    U, D = R.shape

    if Y is not None:
        fix_movies = True
        K, _ = Y.shape
    else:
        Y = 5 * np.random.rand(K, D).astype(np.float64, copy=False)

    W = W.astype(np.float64, copy=False)
    R = R.astype(np.float64, copy=False)
    X = 5 * np.random.rand(U, K).astype(np.float64, copy=False)
    B = get_bias(R, D, U).astype(np.float64, copy=False)
    error_log = []
    error_test_log = []
    _lambda = 0.05

    if biased:
        R = R - B

    if fix_movies:
        for u in range(U):
            Wu = np.diag(W[u])
            X[u] = np.linalg.solve(np.dot(Y, np.dot(Wu, Y.T)) + _lambda * np.eye(K),
                                   np.dot(Y, np.dot(Wu, R[u].T))).T
            err = get_biased_error(R, W, X, Y, B) if biased else get_error(R, W, X, Y)

    err = np.inf
    while steps > 0 and err > 0.002:
        for u in range(U):
            Wu = np.diag(W[u])
            X[u] = np.linalg.solve(np.dot(Y, np.dot(Wu, Y.T)) + _lambda * np.eye(K),
                                   np.dot(Y, np.dot(Wu, R[u].T))).T
        if not fix_movies:
            for i in range(D):
                Wi = np.diag(W.T[i])
                Y[:, i] = np.linalg.solve(np.dot(X.T, np.dot(Wi, X)) + _lambda * np.eye(K),
                                          np.dot(X.T, np.dot(Wi, R[:, i])))

        err = get_biased_error(R, W, X, Y, B) if biased else get_error(R, W, X, Y)
        error_log.append(err)
        print('Error: {}'.format(err))

        if R_test is not None:
            err_test = get_biased_error(R_test, W_test, X, Y, B) if biased else get_error(R_test, W_test, X, Y)
            error_test_log.append(err_test)
            print('Test Error: {}'.format(err_test))

        steps = steps - 1

    plt.plot(error_log)
    if R_test is not None:
        plt.plot(error_test_log, 'r')
    plt.title('Learning RMSE')
    plt.xlabel('Iteration count')
    plt.ylabel('Error')
    plt.show()

    return X, Y, B


def del_ratings_without_meta(ratings, mov_ids):
    return ratings[ratings['movieId'].isin(mov_ids)]


def get_rating_matrix(test_pct=None, mov_ids=None):
    ratings = pd.read_csv('Data/ratings.csv')
    ratings = del_ratings_without_meta(ratings, mov_ids)
    ratings = reduce_ratings(ratings, 500)
    ratings = reduce_ratings(ratings, 1000, by_movie=False)
    num_movies = len(ratings['movieId'].unique())
    num_users = len(ratings['userId'].unique())
    mov_map = map_to_ind(ratings['movieId'].unique())
    user_map = map_to_ind(ratings['userId'].unique())
    r = np.zeros([num_users, num_movies])
    y = np.zeros([num_users, num_movies])
    n_ratings = ratings.shape[0]

    r_test = np.zeros([num_users, num_movies])
    y_test = np.zeros([num_users, num_movies])

    for ri, (idx, rating) in enumerate(ratings.iterrows()):
        i = user_map[int(rating['userId'])]
        j = mov_map[int(rating['movieId'])]

        if mov_ids is not None and j not in mov_ids:
            continue

        if test_pct is not None and float(ri) / float(n_ratings) < test_pct:
            y_test[i][j] = 1
            r_test[i][j] = rating['rating']
        else:
            y[i][j] = 1
            r[i][j] = rating['rating']

    return r, y, mov_map, user_map, r_test, y_test, ratings
