import numpy as np


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


def factorize(R, Y, K, steps=3000, alpha=0.0002, beta=0.02, Q=None):
    """
    :param R: R[x][y] -> user x, item y
    :param K:
    :param steps:
    :param alpha:
    :param beta:
    :param Q:
    :return:
    """
    U = len(R)
    D = len(R[0])
    P = np.random.rand(U, K)  # User factor
    if Q is None:
        Q = np.random.rand(K, D)  # Movies factor
    else:
        K = len(Q[0])

    mu = avg(R)

    bq = item_bias(R, D, U, mu)
    bp = user_bias(R, D, U, mu, bq)  # User bias

    B = np.zeros([U, D])
    for i in range(0, U):
        for j in range(0, D):
            B[i][j] = mu + bq[j] + bp[i]

    R = R - B

    while steps > 0:
        for i in range(0, U):
            for j in range(0, D):
                if Y[i][j] > 0:
                    err = R[i][j] - np.dot(P[i, :], Q[:, j])
                    eij = err
                    for k in range(0, K):
                        P[i][k] = P[i][k] + alpha * (2 * Q[k][j] * eij - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * P[i][k] * eij - beta * Q[k][j])

        e = 0

        for i in range(U):
            for j in range(D):
                if Y[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if steps % 10 == 0:
            print(steps, e)
        if e < 0.001:
            break
        steps = steps - 1

    return P, Q.T, B


def approx():
    R = np.array([[5, 3, 0, 1],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4]])
    Y = np.array([[1, 1, 0, 1],
                  [1, 0, 0, 1],
                  [1, 1, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 1]])

    P, Q, B = factorize(R, Y, 2)

    print((np.dot(P, Q.T)) + B)
