# -*- coding: utf-8 -*-
# @Date    : 2022/1/26 12:20
# @Author  : WangYihao
# @File    : fit_plot.py


import os
import warnings
import argparse

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
        x (ndarray(float)): some x points.
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
        data_x (ndarray(int)): xs of raw data.
        data_col (ndarray(float)): ys of raw data. With the same length as data_x.
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


def ax_plot(ax, x, ys, col_idx):
    """

    Args:
        x (ndarray(int)): xs of raw data.
        ys (dict):
            ys["raw"] (ndarray(float)): ys of raw data.
            ys["fit"] (ndarray(float)): ys of fitted data.
        col_idx (int): index of column.

    Returns:
        None. Plot directly.
    """

    ax.scatter(x, ys["raw"], label=f'raw data {col_idx}',
                c='none', marker='o', s=5, edgecolors=f'C{col_idx}')
    ax.plot(x, ys["fit"], label=f'fit curve {col_idx}', c=f'C{col_idx}', ls='--')

    ax.set_xlim([100, 600])
    ax.set_ylim([0, 1])
    ax.legend(loc='best')


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True,
                    help="the .mat file you want to do gaussian fit.")
args = parser.parse_args()

if __name__ == "__main__":
    # data pre-process
    data_y = np.zeros((2976, 1))
    data_y[:-1] = loadmat(args.file)['val1']
    data_y = data_y.reshape((-1, 4))
    data_x = np.arange(0, len(data_y))

    # fit and plot
    para_names = ['A', 'mu', 'sigma', 'y0']

    # prepare for ploting
    plt.rcParams["figure.figsize"] = (8.0, 6.0)
    plt.rcParams["figure.dpi"] = 200
    fig1, axes1 = plt.subplots(nrows=2, ncols=2, tight_layout=True)
    axes1 = axes1.reshape((4,))
    fig2, axes2 = plt.subplots(nrows=2, ncols=2, tight_layout=True)
    axes2 = axes2.reshape((4,))

    fig_list = [fig1, fig2]

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
        ax_plot(axes1[i], x_fit, {"raw": data_col_fit, "fit": gauss(x_fit, *p0)}, col_idx=i)

        # 拟合
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            try:
                popt, pcov = scipy.optimize.curve_fit(f=gauss, xdata=x_fit, ydata=data_col_fit, p0=p0)
                # 拟合结果参数值
                for name, para in zip(para_names, popt):
                    print(f"{name}:\t{para:.4f}")
                ax_plot(axes2[i], data_x, {"raw": data_col, "fit": gauss(data_x, *popt)}, col_idx=i)
            except (OptimizeWarning, RuntimeError):
                print("**********拟合失败**********\n")

        print()

    fig1.show()
    fig2.show()
    input("Press Enter to close all figures.")
