import os
from sklearn.model_selection import KFold
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
from picture_and_penalty_func import SCAD, HardThresholding
from scad import SCADFit


def model_error(beta, beta_estimate, x):
    '''

    :param beta: 1 * 8
    :param beta_estimate: 1 * 8
    :param x: n * 8
    :return: model error  of page 1355, Fan and Li(2001)
    '''
    return np.matmul(np.matmul(np.expand_dims(beta - beta_estimate, axis=0), np.matmul(x.T, x)),
                     np.expand_dims(beta - beta_estimate, axis=0).T).item()


def generate_data(datasets=100, n=40, sigma=3, rou=0.5, beta=np.array([3, 1.5, 0, 0, 2, 0, 0, 0]), method='normal'):
    cov_ = np.array([1])
    for i in range(1, len(beta)):
        cov_ = np.append(cov_, rou ** i)

    cov_matrix = [cov_.tolist()]
    for i in range(1, len(beta)):
        cov_matrix.append(np.append(np.array([0] * i), cov_[:-i]).tolist())

    # print(cov_matrix)

    for i in range(len(beta)):
        for j in range(i):
            cov_matrix[i][j] = cov_matrix[j][i]

    # print(np.array(cov_matrix).shape)

    X = np.random.multivariate_normal([0] * len(beta), np.array(cov_matrix), size=(datasets, n))

    if method == 'outlier':
        epsilon = np.hstack((np.random.normal(0, 1, size=(datasets, int(0.9 * n))),
                             np.random.standard_cauchy(size=(datasets, n - int(0.9 * n)))))
    else:
        epsilon = np.random.normal(0, 1, size=(datasets, n))

    Y = np.matmul(X, beta) + epsilon * sigma
    return X, Y


def simulation_linear_reg(datasets, n_sigma, rou, beta, method="normal"):
    n = n_sigma[0]
    sigma = n_sigma[1]

    X, Y = generate_data(datasets, n, sigma, rou, beta, method)

    zero_list = [2, 3, 5, 6, 7]
    scad_lambda_params = np.arange(1.5, 4.6, 0.1)
    # scad_lambda_param = np.sqrt(2)
    kf = KFold(n_splits=5)
    model_error_5fold_list = []
    for scad_lambda in scad_lambda_params:
        model_error_list = []
        for train, test in kf.split(X[0]):
            scad = SCADFit(X[0][train], Y[0][train], lambd=scad_lambda)
            scad.fit()
            # mean_squared_error_list.append(mean_squared_error(scad.predict(X[0][test]), Y[0][test]))
            model_error_list.append(model_error(beta, scad.coef, X[0][train]))
        model_error_5fold_list.append(np.mean(model_error_list))
    best_scad_lambda = scad_lambda_params[np.argmin(model_error_5fold_list)]

    print("Best lambda for scad:  ", best_scad_lambda)
    # best_scad_lambda = 2.5

    lasso_beta_estimate_list = []
    lasso_model_error_list = []
    lasso_zero_number_list = []
    lasso_zero_number_error_list = []
    ridge_beta_estimate_list = []
    ridge_model_error_list = []
    ridge_zero_number_list = []
    ridge_zero_number_error_list = []
    ht_beta_estimate_list = []
    ht_model_error_list = []
    ht_zero_number_list = []
    ht_zero_number_error_list = []
    scad_beta_estimate_list = []
    scad_model_error_list = []
    scad_zero_number_list = []
    scad_zero_number_error_list = []
    scad_5fold_beta_estimate_list = []
    scad_5fold_model_error_list = []
    scad_5fold_zero_number_list = []
    scad_5fold_zero_number_error_list = []

    lasso = linear_model.Lasso(alpha=0.1)
    ridge = linear_model.Ridge(alpha=1)
    zero_list = [2, 3, 5, 6, 7]

    filename = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(filename):
        os.makedirs(filename)
    for i in range(0, datasets):
        # print(X[i].shape)
        # print(Y[i].shape)
        lasso.fit(X[i], Y[i])
        lasso_beta_estimate_list.append(lasso.coef_)
        lasso_zero_list = np.where(np.array(lasso.coef_) == 0)[0].tolist()
        lasso_zero_number_list.append(len([x for x in lasso_zero_list if x in zero_list]))
        lasso_zero_number_error_list.append(len(lasso_zero_list) - len([x for x in lasso_zero_list if x in zero_list]))

        ridge.fit(X[i], Y[i])
        ridge_beta_estimate_list.append(ridge.coef_)
        ridge_zero_list = np.where(np.array(ridge.coef_) == 0)[0].tolist()
        ridge_zero_number_list.append(len([x for x in ridge_zero_list if x in zero_list]))
        ridge_zero_number_error_list.append(len(ridge_zero_list) - len([x for x in ridge_zero_list if x in zero_list]))

        scad_lambda_param = 2
        # scad_lambda_param = np.sqrt(2)
        scad = SCADFit(X[i], Y[i], lambd=scad_lambda_param)
        scad.fit()
        scad_beta_estimate_list.append(scad.coef)
        scad_zero_list = np.where(np.array(scad.coef) == 0)[0].tolist()
        scad_zero_number_list.append(len([x for x in scad_zero_list if x in zero_list]))
        scad_zero_number_error_list.append(len(scad_zero_list) - len([x for x in scad_zero_list if x in zero_list]))

        scad_5fold = SCADFit(X[i], Y[i], lambd=best_scad_lambda)
        scad_5fold.fit()
        scad_5fold_beta_estimate_list.append(scad_5fold.coef)
        scad_5fold_zero_list = np.where(np.array(scad_5fold.coef) == 0)[0].tolist()
        scad_5fold_zero_number_list.append(len([x for x in scad_5fold_zero_list if x in zero_list]))
        scad_5fold_zero_number_error_list.append(
            len(scad_5fold_zero_list) - len([x for x in scad_5fold_zero_list if x in zero_list]))

        ht_lambda_param = 2
        ht = HardThresholding(tau=ht_lambda_param)
        ht_coef = ht.fit(X[i], Y[i])
        ht_beta_estimate_list.append(ht_coef)
        ht_zero_list = np.where(np.array(ht_coef) < 0.001)[0].tolist()
        ht_zero_number_list.append(len([x for x in ht_zero_list if x in zero_list]))
        ht_zero_number_error_list.append(len(ht_zero_list) - len([x for x in ht_zero_list if x in zero_list]))

        lasso_model_error_list.append(model_error(beta, lasso.coef_, X[i]))
        ridge_model_error_list.append(model_error(beta, ridge.coef_, X[i]))
        scad_model_error_list.append(model_error(beta, scad.coef, X[i]))
        ht_model_error_list.append(model_error(beta, ht_coef, X[i]))
        # break

    with open("./results/results_{}_{}_{}_{}.txt".format(datasets, n, sigma, method), 'w') as f:
        f.write("lasso MRME  " + str(np.median(lasso_model_error_list)) + '\n')
        f.write("Ridge MRME  " + str(np.median(ridge_model_error_list)) + '\n')
        f.write("SCAD MRME  " + str(np.median(scad_model_error_list)) + '\n')
        f.write("SCAD 5-fold MRME  " + str(np.median(scad_5fold_model_error_list)) + '\n')
        f.write("Hard Threshold MRME  " + str(np.median(ht_model_error_list)) + '\n')
        f.write("lasso Correct  " + str(np.mean(lasso_zero_number_list)) + '\n')
        f.write("Ridge Correct  " + str(np.mean(ridge_zero_number_list)) + '\n')
        f.write("SCAD Correct  " + str(np.mean(scad_zero_number_list)) + '\n')
        f.write("SCAD 5-fold Correct  " + str(np.mean(scad_5fold_zero_number_list)) + '\n')
        f.write("Hard Threshold Correct  " + str(np.mean(ht_zero_number_list)) + '\n')
        f.write("lasso InCorrect  " + str(np.mean(lasso_zero_number_error_list)) + '\n')
        f.write("Ridge InCorrect  " + str(np.mean(ridge_zero_number_error_list)) + '\n')
        f.write("SCAD InCorrect  " + str(np.mean(scad_zero_number_error_list)) + '\n')
        f.write("SCAD 5-fold InCorrect  " + str(np.mean(scad_5fold_zero_number_error_list)) + '\n')
        f.write("Hard Threshold InCorrect  " + str(np.mean(ht_zero_number_error_list)) + '\n')

    with open("./results/LASSO_estimate_{}_{}_{}_{}.txt".format(datasets, n, sigma, method), 'w') as f:
        f.write(str(lasso_beta_estimate_list) + '\n')
    with open("./results/Ridge_estimate_{}_{}_{}_{}.txt".format(datasets, n, sigma, method), 'w') as f:
        f.write(str(ridge_beta_estimate_list) + '\n')
    with open("./results/Hard Threshold_estimate_{}_{}_{}_{}.txt".format(datasets, n, sigma, method), 'w') as f:
        f.write(str(ht_beta_estimate_list) + '\n')
    with open("./results/SCAD_estimate_{}_{}_{}_{}.txt".format(datasets, n, sigma, method), 'w') as f:
        f.write(str(scad_beta_estimate_list) + '\n')
    with open("./results/SCAD_5fold_estimate_{}_{}_{}_{}.txt".format(datasets, n, sigma, method), 'w') as f:
        f.write("Best lambda for scad:  " + str(best_scad_lambda) + '\n')
        f.write(str(scad_beta_estimate_list) + '\n')


if __name__ == '__main__':
    np.random.seed(2)
    datasets = 100
    # n = 40
    # sigma = 3
    rou = 0.5
    beta = np.array([3, 1.5, 0, 0, 2, 0, 0, 0])
    for n_sigma in [[40, 3], [40, 1], [60, 1]]:
        simulation_linear_reg(datasets, n_sigma, rou, beta)
    for n_sigma in [[60, 1]]:
        simulation_linear_reg(datasets, n_sigma, rou, beta, method='outlier')

# print("lasso MRME", np.median(lasso_model_error_list))
# print("Ridge MRME", np.median(ridge_model_error_list))
# print("SCAD MRME", np.median(scad_model_error_list))
# print("Hard Threshold MRME", np.median(ht_model_error_list))
# print("lasso Correct", np.mean(lasso_zero_number_list))
# print("Ridge Correct", np.mean(ridge_zero_number_list))
# print("SCAD Correct", np.mean(scad_zero_number_list))
# print("Hard Threshold Correct", np.mean(ht_zero_number_list))
# print("lasso InCorrect", np.mean(lasso_zero_number_error_list))
# print("Ridge InCorrect", np.mean(ridge_zero_number_error_list))
# print("SCAD InCorrect", np.mean(scad_zero_number_error_list))
# print("Hard Threshold InCorrect", np.mean(ht_zero_number_error_list))
