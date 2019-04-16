# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def data_load(path):
    """
    Load dataset.

    :param path: Path to folder with *.txt files
    :return:
        lst_id  - List of id:
        lst_vec - List of bio-models - np.array
    """
    lst_id = []
    lst_vec = []
    for fname in glob.iglob(path + "/*.txt"):
        with open(fname, "r") as fi:
            line = fi.readlines()
        data = np.fromstring(line[0].replace('[', '').replace(']', ''), dtype=np.float32, sep=' ')

        lst_id.append(os.path.splitext(os.path.basename(fname))[0])
        lst_vec.append(data)

    print("Dataset {} loaded.".format(path))
    return lst_id, lst_vec


def compare_id(id_a, id_b):
    """
    Compare id.

    :param id_a: - str
    :param id_b: - str
    :return:
    """
    return id_a[:4] == id_b[:4]


def compare_ivec(ivec_a, ivec_b):
    """
    Function takes  2 bio models and compares them.


    :param ivec_a: - np.array
    :param ivec_b: - np.array
    :return:
    """
    x = ivec_a.reshape(1, -1)
    y = ivec_b.reshape(1, -1)
    return cosine_similarity(x, y)[0, 0]


def calc_scores(ids, ivecs):
    """
    Function compare each other all bio-models

    :param ids: - List of id:
    :param ivecs: - List of bio-models - np.array
    :return:
        lst_compare_key_result - List of compare id: True - equal, False - otherwise.
        lst_compare_ivec_result - List of scores
    """

    map_id_ivec = {k: v for k, v in zip(ids, ivecs)}

    lst_compare_key_result = []
    lst_compare_ivec_result = []

    for i, comp in tqdm(enumerate(itertools.combinations(ids, 2))):
        eq_id = compare_id(*comp)
        score = compare_ivec(map_id_ivec[comp[0]], map_id_ivec[comp[1]])

        lst_compare_key_result.append(eq_id)
        lst_compare_ivec_result.append(score)

    # with open("./tmp/scores.pickle", "wb") as pk:
    #     pickle.dump((lst_compare_key_result, lst_compare_ivec_result), pk)

    return lst_compare_key_result, lst_compare_ivec_result


def whiten(X, fudge=1E-18):
    """
    Whitening X.
    The matrix X should be observations-by-components

    :param X:
    :param fudge:
    :return:
    """

    # get the covariance matrix
    Xcov = np.dot(X.T, X)

    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)

    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1. / np.sqrt(d+fudge))

    # whitening matrix
    W = np.dot(np.dot(V, D), V.T)

    # multiply by the whitening matrix
    X_white = np.dot(X, W)

    return X_white, W


def plot_fr_fa(lst_compare_key_result, lst_compare_ivec_result):
    """
    Plot FR/FA curve

    :param lst_compare_key_result:
    :param lst_compare_ivec_result:
    :return:
    """

    # with open("./tmp/scores.pickle", "rb") as pk:
    #     (lst_compare_key_result, lst_compare_ivec_result) = pickle.load(pk)

    positive = sum(lst_compare_key_result)
    negative = len(lst_compare_key_result) - positive

    lst_compare_ivec_result, lst_compare_key_result = zip(*sorted(zip(lst_compare_ivec_result, lst_compare_key_result)))

    FA = negative
    FR = 0
    curve_FR = []
    curve_FA = []

    for i, _ in enumerate(lst_compare_ivec_result):
        if lst_compare_key_result[i] == False:
            FA -= 1
        else:
            FR += 1
        curve_FR.append(FR)
        curve_FA.append(FA)

    curve_FR = np.array(curve_FR) / positive
    curve_FA = np.array(curve_FA) / negative

    # Plot
    fig, ax = plt.subplots()
    ax.plot(lst_compare_ivec_result, curve_FR, label='FR')
    ax.plot(lst_compare_ivec_result, curve_FA, label='FA')
    ax.legend(loc="best")

    ax.set(xlabel='Score', ylabel='Probability',
           title='FR/FA curve')
    ax.grid()

    fig.savefig("fr_fa.png")
    plt.show()


def plot_hist_scores(lst_compare_key_result, lst_compare_ivec_result):
    """
    Plot hist scores

    :param lst_compare_key_result:
    :param lst_compare_ivec_result:
    :return:
    """

    y = filter(lambda x: x[0], zip(lst_compare_key_result, lst_compare_ivec_result))
    p_scores = list(list(zip(*y))[1])

    y = filter(lambda x: not x[0], zip(lst_compare_key_result, lst_compare_ivec_result))
    n_scores = list(list(zip(*y))[1])

    # Plot histogram of the data

    plt.hist(p_scores, 100, normed=1, facecolor='green', alpha=0.75)
    plt.hist(n_scores, 500, normed=1, facecolor='red', alpha=0.75)

    plt.xlabel('Scores')
    plt.ylabel('Normed hist (PDF)')
    plt.title('Histogram of scores (green - equal person, red - otherwise)')
    plt.grid(True)

    plt.show()





