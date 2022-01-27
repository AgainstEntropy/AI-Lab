# -*- coding: utf-8 -*-
# @Date    : 2022/1/26 12:20
# @Author  : WangYihao
# @File    : fit_plot.py


import os

import numpy as np
import scipy.optimize
from scipy.optimize import OptimizeWarning
from scipy.io import loadmat
import matplotlib.pyplot as plt


def gauss(x, A, mu, sigma, y0):
    """
    A gaussian function is like:
        A * exp((x-mu)^2 / 2 * sigma^2) + y0
    Args:
        x (list(float)): some x points.
        A (float): Amplitude.
        mu (float): Mean.
        sigma (float): Variance.
        y0 (float): Bias.

    Returns:

    """
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + y0


def data_filter(data_x, data_col, col_idx, threshold=0.25, verbose=True):
    """

    Args:
        data_x (list(int)): xs of raw data.
        data_col (list(float)): ys of raw data. With the same length as data_x.
        threshold (float): Set threshold for filter out values away from gaussian peak.

    Returns:
        x_fit, data_col_fit: data reserved for fitting directly.

    """
    # filter out zero value
    data_col_fit_idx = data_col != 0.
    x_fit = data_x[data_col_fit_idx]
    data_col_fit = data_col[data_col_fit_idx]

    # filter out values away from gaussian peak
    data_col_fit_idx = \
        data_col_fit > min(data_col_fit) + (max(data_col_fit) - min(data_col_fit)) * threshold
    data_col_fit = data_col_fit[data_col_fit_idx]
    x_fit = x_fit[data_col_fit_idx]

    if verbose:
        print(f"number of points used for fitting column {col_idx}: {len(x_fit)}")

    if len(x_fit) < 55:
        raise ValueError

    return x_fit, data_col_fit


def plot(x, ys, col_idx):
    """

    Args:
        x (list(int)): xs of raw data.
        ys (dict):
            ys["raw"] (list(float)): ys of raw data.
            ys["fit"] (list(float)): ys of fitted data.
        col_idx (int): index of column.

    Returns:
        None. Plot directly.
    """

    plt.scatter(x, ys["raw"], label=f'origin data {col_idx}(used for fitting)',
                c='none', marker='o', s=5, edgecolors=f'C{col_idx}')
    plt.plot(x, ys["fit"], label=f'fit curve {col_idx}', c=f'C{col_idx}', ls='--')

    plt.xlim([100, 600])
    plt.ylim([0, 1])
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    # data pre-process
    file_path = './mats/149.mat'
    data_y = np.zeros((2976, 1))
    data_y[:-1] = loadmat(file_path)['val1']
    data_y = data_y.reshape((-1, 4))
    data_x = np.arange(0, len(data_y))

    # fit and plot
    para_names = ['A', 'mu', 'sigma', 'y0']
    # 初值
    p0 = np.array([0.5, 300., 20, 0.1])
    for i in range(4):
        data_col = data_y[:, i]

        # 筛选用于拟合的点集
        try:
            x_fit, data_col_fit = data_filter(data_x, data_col, i, threshold=0.25)
        except ValueError:
            print("*****数据过少,无法用于拟合*****\n")
            continue

        # 拟合
        plot(x_fit, {"raw": data_col_fit, "fit": gauss(x_fit, *p0)}, col_idx=i)

        # 拟合
        try:
            popt, pcov = scipy.optimize.curve_fit(f=gauss, xdata=x_fit, ydata=data_col_fit, p0=p0)
            # 拟合结果参数值
            for name, para in zip(para_names, popt):
                print(f"{name}:\t{para:.4f}")
            plot(data_x, {"raw": data_col, "fit": gauss(data_x, *popt)}, col_idx=i)
        except (OptimizeWarning, RuntimeError):
            # TODO: OptimizeWarning...
            print("**********拟合失败**********\n")

        print()
