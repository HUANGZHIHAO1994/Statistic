import os
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

from optim import Adam


def _check_wts(weights, wts):
    """helper function for deprecating `wts`
    """
    if wts is not None:
        import warnings
        warnings.warn('`wts` method is deprecated. Use `weights` instead',
                      DeprecationWarning)
    weights = weights if weights is not None else wts
    return weights


class Penalty(object):
    """
    A class for representing a scalar-value penalty.
    Parameters
    ----------
    weights : array-like
        A vector of weights that determines the weight of the penalty
        for each parameter.
    Notes
    -----
    The class has a member called `alpha` that scales the weights.
    """

    def __init__(self, weights):
        self.weights = weights
        self.alpha = 1.

    def func(self, params):
        """
        A penalty function on a vector of parameters.
        Parameters
        ----------
        params : array-like
            A vector of parameters.
        Returns
        -------
        A scalar penaty value; greater values imply greater
        penalization.
        """
        raise NotImplementedError

    def deriv(self, params):
        """
        The gradient of a penalty function.
        Parameters
        ----------
        params : array-like
            A vector of parameters
        Returns
        -------
        The gradient of the penalty with respect to each element in
        `params`.
        """
        raise NotImplementedError

    def grad(self, params):
        import warnings
        warnings.warn('grad method is deprecated. Use `deriv` instead',
                      DeprecationWarning)
        return self.deriv(params)


class L1(Penalty):
    """
    The L1 (LASSO) penalty.
    """

    def __init__(self, weights=None, wts=None):
        weights = _check_wts(weights, wts)  # for deprecation wts
        if weights is None:
            self.weights = 1.
        else:
            self.weights = weights
        self.alpha = 1.

    def func(self, params):
        return np.sum(self.weights * self.alpha * np.abs(params)), self.weights * self.alpha * np.abs(params)

    def deriv(self, params):
        p = np.atleast_1d(params)
        p_abs = np.abs(p)
        res = np.empty(p_abs.shape)
        res.fill(np.nan)

        mask1 = p != 0
        mask2 = ~mask1
        res[mask1] = self.weights * self.alpha
        res[mask2] = 0
        return res

    def estimate(self, z):
        return np.sign(z) * np.maximum(np.abs(z) - self.alpha, 0)


class L2(Penalty):
    """
    The L2 (ridge) penalty.
    """

    def __init__(self, weights=None, wts=None):
        weights = _check_wts(weights, wts)  # for deprecation wts
        if weights is None:
            self.weights = 1.
        else:
            self.weights = weights
        self.alpha = 1.

    def func(self, params):
        return np.sum(self.weights * self.alpha * params ** 2), self.weights * self.alpha * params ** 2

    def deriv(self, params):
        return 2 * self.weights * self.alpha * params

    def deriv2(self, params):
        return 2 * self.weights * self.alpha * np.ones(len(params))


class Lq(Penalty):
    """
    The Lq (bridge) penalty.
    """

    def __init__(self, weights=None, wts=None, q=2.):
        weights = _check_wts(weights, wts)  # for deprecation wts
        if weights is None:
            self.weights = 1.
        else:
            self.weights = weights
        self.alpha = 1.
        self.q = q

    def func(self, params):
        return np.sum(self.weights * self.alpha * np.power(params, self.q)), self.weights * self.alpha * np.power(
            params, self.q)

    def deriv(self, params):
        p = np.atleast_1d(params)
        p_abs = np.abs(p)
        res = np.empty(p_abs.shape)
        res.fill(np.nan)

        mask1 = p != 0
        mask2 = ~mask1
        res[mask1] = self.q * self.alpha * np.power(
            params, self.q - 1)
        res[mask2] = 0
        return res


class HardThresholding(Penalty):
    """
    The HardThresholding penalty.
    Fan and Li use lambda instead of tau. Fan and Li
    """

    def __init__(self, tau=2, weights=None):
        if weights is None:
            self.weights = 1.
        else:
            self.weights = weights
        self.tau = tau

    def func(self, params):

        # 2 segments in absolute value
        tau = self.tau
        p_abs = np.atleast_1d(np.abs(params))
        res = np.empty(p_abs.shape, p_abs.dtype)
        res.fill(np.nan)
        mask1 = p_abs >= tau
        res[mask1] = tau ** 2
        mask2 = ~mask1
        p_abs2 = p_abs[mask2]
        res[mask2] = tau ** 2 - (p_abs2 - tau) ** 2
        return (self.weights * res).sum(0), res

    def deriv(self, params):

        # 2 segments in absolute value
        tau = self.tau
        p = np.atleast_1d(params)
        p_abs = np.abs(p)
        p_sign = np.sign(p)
        res = np.empty(p_abs.shape)
        res.fill(np.nan)

        mask1 = p_abs < tau
        mask2 = ~mask1
        res[mask1] = -2 * p_sign[mask1] * (p_abs[mask1] - tau)
        res[mask2] = 0

        return self.weights * res

    def estimate(self, z):
        # 2 segments in absolute value
        tau = self.tau
        z = np.atleast_1d(z)
        z_abs = np.abs(z)
        res = np.empty(z_abs.shape)
        res.fill(np.nan)

        mask1 = z_abs < np.sqrt(2) * tau
        mask2 = ~mask1
        res[mask1] = 0
        res[mask2] = z[mask2]

        return self.weights * res

    def fit(self, x, y, iter=100000, lr=0.003):
        ols = LinearRegression().fit(x, y)
        self.coef = ols.coef_
        adam = Adam()
        for i in range(iter):
            # if i % 1000 == 0:
            #     print(self.coef)
            grad = -np.matmul(x.T, y - np.matmul(x, self.coef)) + self.deriv(self.coef)
            adam(self.coef, grad)
        # print(self.deriv(self.coef))
        # print(np.matmul(x.T, y - np.matmul(x, self.coef)))

        # GD
        # for i in range(iter):
        #     grad = -np.matmul(x.T, y - np.matmul(x, self.coef)) + self.deriv(self.coef)
        #     if i % 1000 == 0:
        #         print(self.coef)
        #     self.coef -= lr * grad
        return self.coef


class SCAD(Penalty):
    """
    The SCAD penalty of Fan and Li.
    The SCAD penalty is linear around zero as a L1 penalty up to threshold tau.
    The SCAD penalty is constant for values larger than c*tau.
    The middle segment is quadratic and connect the two segments with a continuous
    derivative.
    The penalty is symmetric around zero.
    Parameterization follows Boo, Johnson, Li and Tan 2011.
    Fan and Li use lambda instead of tau, and a instead of c. Fan and Li
    recommend setting c=3.7.
    f(x) = { tau |x|                                        if 0 <= |x| < tau
           { -(|x|^2 - 2 c tau |x| + tau^2) / (2 (c - 1))   if tau <= |x| < c tau
           { (c + 1) tau^2 / 2                              if c tau <= |x|
    Parameters
    ----------
    tau : float
        slope and threshold for linear segment
    c : float
        factor for second threshold which is c * tau
    weights : None or array
        weights for penalty of each parameter. If an entry is zero, then the
        corresponding parameter will not be penalized.
    References
    ----------
    Buu, Anne, Norman J. Johnson, Runze Li, and Xianming Tan. "New variable
    selection methods for zero‐inflated count data with applications to the
    substance abuse field."
    Statistics in medicine 30, no. 18 (2011): 2326-2340.
    Fan, Jianqing, and Runze Li. "Variable selection via nonconcave penalized
    likelihood and its oracle properties."
    Journal of the American statistical Association 96, no. 456 (2001):
    1348-1360.
    """

    def __init__(self, tau, c=3.7, weights=None):
        if weights is None:
            self.weights = 1.
        else:
            self.weights = weights
        self.tau = tau
        self.c = c

    def func(self, params):

        # 3 segments in absolute value
        tau = self.tau
        p_abs = np.atleast_1d(np.abs(params))
        res = np.empty(p_abs.shape, p_abs.dtype)
        res.fill(np.nan)
        mask1 = p_abs < tau
        mask3 = p_abs >= self.c * tau
        res[mask1] = tau * p_abs[mask1]
        mask2 = ~mask1 & ~mask3
        p_abs2 = p_abs[mask2]
        tmp = (p_abs2 ** 2 - 2 * self.c * tau * p_abs2 + tau ** 2)
        res[mask2] = -tmp / (2 * (self.c - 1))
        res[mask3] = (self.c + 1) * tau ** 2 / 2.
        return (self.weights * res).sum(0), res

    def deriv(self, params):

        # 3 segments in absolute value
        tau = self.tau
        p = np.atleast_1d(params)
        p_abs = np.abs(p)
        p_sign = np.sign(p)
        res = np.empty(p_abs.shape)
        res.fill(np.nan)

        mask1 = p_abs < tau
        mask3 = p_abs >= self.c * tau
        mask2 = ~mask1 & ~mask3
        res[mask1] = p_sign[mask1] * tau
        tmp = p_sign[mask2] * (p_abs[mask2] - self.c * tau)
        res[mask2] = -tmp / (self.c - 1)
        res[mask3] = 0

        return self.weights * res

    def deriv2(self, params):
        """Second derivative of function
        This returns scalar or vector in same shape as params, not a square
        Hessian. If the return is 1 dimensional, then it is the diagonal of
        the Hessian.
        """

        # 3 segments in absolute value
        tau = self.tau
        p = np.atleast_1d(params)
        p_abs = np.abs(p)
        res = np.zeros(p_abs.shape)

        mask1 = p_abs < tau
        mask3 = p_abs >= self.c * tau
        mask2 = ~mask1 & ~mask3
        res[mask2] = -1 / (self.c - 1)

        return self.weights * res

    def estimate(self, z):
        """
        optimal estimation
        :return:
        """
        # 3 segments in absolute value
        tau = self.tau
        z = np.atleast_1d(z)
        z_abs = np.abs(z)
        z_sign = np.sign(z)
        res = np.zeros(z_abs.shape)

        mask1 = z_abs <= 2 * tau
        mask3 = z_abs > self.c * tau
        mask2 = ~mask1 & ~mask3
        res[mask1] = z_sign[mask1] * np.maximum(z_abs[mask1] - tau, 0)
        res[mask2] = ((self.c - 1) * z[mask2] - z_sign[mask2] * self.c * tau) / (self.c - 2)
        res[mask3] = z[mask3]
        return self.weights * res


def fig1():
    plt.style.use('seaborn-dark')

    fig, ax = plt.subplots()

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    param = np.arange(-5, 5, 0.01)
    # print(param)
    # for i in [2, 8, 10]:
    #     # lambda_param = 2
    #     lambda_param = i
    #
    #     # print(param)
    #     scad = SCAD(tau=lambda_param)
    #
    #     _, scad_penalty = scad.func(params=param)
    #
    #     # print(penalty)
    #
    #     ax.plot(param, scad_penalty, label='SCAD_{}'.format(i))

    scad_lambda_param = np.sqrt(2 * np.log(1.7))
    # scad_lambda_param = np.sqrt(2)
    scad = SCAD(tau=scad_lambda_param)
    _, scad_penalty = scad.func(params=param)
    ax.plot(param, scad_penalty, label='SCAD_{}'.format(scad_lambda_param))

    l2 = L2()
    l2.alpha = 0.2
    _, l2_penalty = l2.func(param)
    ax.plot(param, l2_penalty, label='L2')

    l1 = L1()
    l1.alpha = 1
    _, l1_penalty = l1.func(param)
    ax.plot(param, l1_penalty, label='L1')

    ht_lambda_param = 2
    ht = HardThresholding(tau=ht_lambda_param)
    _, ht_penalty = ht.func(params=param)
    ax.plot(param, ht_penalty, label='Hard Thresholding')

    # ax.plot(period, portfolio, label='portfolio')
    ax.set(xlabel='theta', ylabel='penalty',
           title="fig1")

    ax.grid()
    ax.legend()
    plt.savefig("./images/fig1", dpi=300)
    # plt.show()


def fig3():
    plt.style.use('seaborn-dark')

    fig, ax = plt.subplots()

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    scad_lambda_param = 2
    scad = SCAD(tau=scad_lambda_param)

    ht_lambda_param = 2
    ht = HardThresholding(tau=ht_lambda_param)

    param2 = np.arange(0.1, 10, 0.1)
    scad_driv = scad.deriv(params=param2)
    ax.plot(param2, scad_driv + param2, label='theta + SCAD_driv')
    ht_driv = ht.deriv(params=param2)
    ax.plot(param2, ht_driv + param2, label='theta + ht_driv')

    ax.plot(param2, np.ones(param2.shape), label='z=1')
    ax.plot(param2, 3 * np.ones(param2.shape), label='z=3')

    # ax.plot(period, portfolio, label='portfolio')
    ax.set(xlabel='theta', ylabel='theta + penalty_driv',
           title="fig3")

    ax.grid()
    ax.legend()
    # plt.show()
    plt.savefig("./images/fig3", dpi=300)


def fig2():
    plt.style.use('seaborn-dark')

    fig, ax = plt.subplots()

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    z = np.arange(-10, 10, 0.01)

    scad_lambda_param = 2
    # scad_lambda_param = np.sqrt(2)
    scad = SCAD(tau=scad_lambda_param)
    scad_estimate = scad.estimate(z)
    ax.plot(z, scad_estimate, label='SCAD_{}'.format(scad_lambda_param))

    l1 = L1()
    l1.alpha = 1
    l1_estimate = l1.estimate(z)
    ax.plot(z, l1_estimate, label='L1')

    ht_lambda_param = 2
    ht = HardThresholding(tau=ht_lambda_param)
    ht_estimate = ht.estimate(z)
    ax.plot(z, ht_estimate, label='Hard Thresholding')

    # ax.plot(period, portfolio, label='portfolio')
    ax.set(xlabel='z', ylabel='theta_hat',
           title="fig2")

    ax.grid()
    ax.legend()
    # plt.show()
    plt.savefig("./images/fig2", dpi=300)


def fig2_subplot():
    plt.style.use('seaborn-dark')

    fig, ax = plt.subplots(3, 1, sharex='all', sharey='all', figsize=(15, 20))

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    z = np.arange(-10, 10, 0.01)

    scad_lambda_param = 2
    # scad_lambda_param = np.sqrt(2)
    scad = SCAD(tau=scad_lambda_param)
    scad_estimate = scad.estimate(z)
    ax[0].plot(z, scad_estimate, label='SCAD_{}'.format(scad_lambda_param))
    ax[0].set(xlabel='z', ylabel='theta_hat', title="scad")
    ax[0].grid()
    ax[0].legend()

    l1 = L1()
    l1.alpha = 1
    l1_estimate = l1.estimate(z)
    ax[1].plot(z, l1_estimate, label='L1')
    ax[1].set(xlabel='z', ylabel='theta_hat', title="L1")
    ax[1].grid()
    ax[1].legend()

    ht_lambda_param = 2
    ht = HardThresholding(tau=ht_lambda_param)
    ht_estimate = ht.estimate(z)
    ax[2].plot(z, ht_estimate, label='Hard Thresholding')
    ax[2].set(xlabel='z', ylabel='theta_hat', title="Hard Thresholding")
    ax[2].grid()
    ax[2].legend()

    # ax.plot(period, portfolio, label='portfolio')

    # plt.show()
    plt.tight_layout()
    plt.savefig("./images/fig2_subplot", dpi=300)


def fig4():
    plt.style.use('seaborn-dark')

    fig, ax = plt.subplots()

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    scad_lambda_param = 1
    scad = SCAD(tau=scad_lambda_param)

    ht_lambda_param = 1
    ht = HardThresholding(tau=ht_lambda_param)

    param2 = np.arange(0.1, 5, 0.01)
    scad_driv = scad.deriv(params=param2)
    ax.plot(param2, scad_driv, label='SCAD_driv')
    ht_driv = ht.deriv(params=param2)
    ax.plot(param2, ht_driv, label='ht_driv')
    l2 = L2()
    l2.alpha = 1
    l2_driv = l2.deriv(param2)
    ax.plot(param2, l2_driv, label='L2_driv')

    l1 = L1()
    l1.alpha = 1
    l1_driv = l1.deriv(param2)
    ax.plot(param2, l1_driv, label='L1_driv')

    lq = Lq(q=0.5)
    lq.alpha = 1
    lq_driv = lq.deriv(param2)
    ax.plot(param2, lq_driv, label='L0.5_driv')

    # ax.plot(period, portfolio, label='portfolio')
    ax.set(xlabel='theta', ylabel='penalty_driv',
           title="fig4")

    ax.grid()
    ax.legend()
    plt.savefig("./images/fig4", dpi=300)
    # plt.show()


if __name__ == '__main__':
    filename = os.path.join(os.getcwd(), 'images')
    if not os.path.exists(filename):
        os.makedirs(filename)

    fig1()
    fig3()
    fig2()
    fig2_subplot()
    fig4()
