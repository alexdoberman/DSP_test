# -*- coding: utf-8 -*-
import numpy as np
import utils


def main(train_db, test_db):

    # Load train dataset
    _, lst_vec_train = utils.data_load(train_db)

    # Get PCA transform matrix from train data
    X_train = np.array(lst_vec_train)
    _, W = utils.whiten(X_train)

    # Load test dataset
    lst_id, lst_vec_test = utils.data_load(test_db)

    # Whitening test dataset
    X_test = np.array(lst_vec_test)
    X_test_white = np.dot(X_test, W)

    # Calc scores for test dataset
    lst_white_vec = [X_test_white[i, :] for i in range(X_test_white.shape[0])]
    lst_compare_key_result, lst_compare_ivec_result = utils.calc_scores(lst_id, lst_white_vec)

    # Plot FR/FA curve
    utils.plot_fr_fa(lst_compare_key_result, lst_compare_ivec_result)

    # Plot scores hist
    utils.plot_hist_scores(lst_compare_key_result, lst_compare_ivec_result)


if __name__ == '__main__':

    train_db = r"./data/train_db"
    test_db = r"./data/test_db"

    main(train_db, test_db)
