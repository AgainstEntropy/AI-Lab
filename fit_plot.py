# -*- coding: utf-8 -*-
# @Date    : 2022/1/26 12:20
# @Author  : WangYihao
# @File    : fit_plot.py


import os

import numpy as np
import scipy.optimize
from scipy.io import loadmat
import matplotlib.pyplot as plt


def gauss(x, A, mu, sigma, y0):
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + y0


if __name__ == "__main__":
    data_y = np.zeros((2976, 1))
    data_y[:-1] = loadmat('./mats/268.mat')['val1']
    data_y = data_y.reshape((-1, 4))
    data_x = np.arange(0, len(data_y))
    data_col = data_y[:, 3]

    # 筛选用于拟合的点集
    data_col_fit_idx = data_col != 0.
    data_col_fit = data_col[data_col_fit_idx]
    x_fit = data_x[data_col_fit_idx]
    print(x_fit.shape)
    data_col_fit_idx = data_col_fit > min(data_col_fit) + (max(data_col_fit) - min(data_col_fit)) / 4
    data_col_fit = data_col_fit[data_col_fit_idx]
    x_fit = x_fit[data_col_fit_idx]
    print(x_fit.shape)

    # 拟合初值
    p0 = np.array([1., 300., 20, 0.1])
    plt.scatter(x_fit, data_col_fit, label='origin data (used for fitting)', c='none', marker='o', s=5, edgecolors='C0')
    plt.plot(x_fit, gauss(x_fit, *p0), label='initial fit curve', c='C3')
    plt.xlim([100, 600])
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

    # 拟合
    popt, pcov = scipy.optimize.curve_fit(f=gauss, xdata=x_fit, ydata=data_col_fit, p0=p0)
    print('Paras:\n', popt)  # 拟合结果参数值
    print('Cov:\n', pcov)  # 拟合结果协方差

    plt.plot(data_x, gauss(data_x, *popt), c='C3', label='fit curve')
    for i in range(4):
        # plt.plot(data_x, data_y[:, i], label=f'origin data {i}', ls='--')
        plt.scatter(data_x, data_y[:, i], label=f'origin data {i}', c='none', marker='o', s=5, edgecolors=f'C{i}')
    plt.xlim([100, 600])
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
