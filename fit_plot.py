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


def data_filter(data_x, data_col, col_idx, output_file, threshold=0.25, verbose=True):
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
        output_file.write(f"number of points used for fitting column {col_idx}: {len(x_fit)}\n")

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


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--file', required=True,
                    help="the .mat file you want to apply gaussian fit.")
parser.add_argument('-o', '--output', default='./results/',
                    help="output directory")
parser.add_argument('-th', '--threshold', default=0.25, type=float,
                    help="a float threshold for filter out values away from gaussian peak.")
args = parser.parse_args()

if __name__ == "__main__":
    # parse_args
    file_path = args.file
    file_name = file_path.split('/')[-1]
    output_dir = args.output
    threshold = args.threshold

    output_dir = os.path.join(output_dir, file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_txt_path = os.path.join(output_dir, f"{file_name}.txt")
    if os.path.exists(output_txt_path):
        os.remove(output_txt_path)
    output_txt = open(output_txt_path, 'a+')

    # data pre-process
    data_y = np.zeros((2976, 1))
    data_y[:-1] = loadmat(file_path)['val1']
    data_y = data_y.reshape((-1, 4))
    data_x = np.arange(0, len(data_y))

    # fit and plot
    para_names = ['A', 'mu', 'sigma', 'y0']

    # prepare for ploting
    plt.rcParams["figure.figsize"] = (8.0, 6.0)
    plt.rcParams["figure.dpi"] = 200
    fig0 = plt.figure()
    for i in range(4):
        plt.plot(data_y[:, i], c=f'C{i}', label=f'{i}')
    plt.title('origin data')
    plt.legend(loc='best')
    fig0.savefig(output_dir + f'/{file_name}_raw.png')
    fig1, axes1 = plt.subplots(nrows=2, ncols=2, tight_layout=True)
    axes1 = axes1.reshape((4,))
    fig2, axes2 = plt.subplots(nrows=2, ncols=2, tight_layout=True)
    axes2 = axes2.reshape((4,))

    # fig_list = [fig1, fig2]

    # 初值
    p0 = np.array([0.5, 300., 20, 0.1])
    for i in range(4):
        data_col = data_y[:, i]

        # 筛选用于拟合的点集
        try:
            x_fit, data_col_fit = data_filter(data_x, data_col, i, output_txt, threshold=threshold)
        except ValueError:
            print("*****数据过少,无法用于拟合*****\n")
            output_txt.write("*****数据过少,无法用于拟合*****\n\n")
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
                    output_txt.write(f"{name}:\t{para:.4f}\n")
                output_txt.write("\n")
                ax_plot(axes2[i], data_x, {"raw": data_col, "fit": gauss(data_x, *popt)}, col_idx=i)
            except (OptimizeWarning, RuntimeError):
                print("**********拟合失败**********\n")
                output_txt.write("**********拟合失败**********\n\n")

        print()

    output_txt.close()
    fig1.savefig(output_dir + f'/{file_name}1.png')
    fig2.savefig(output_dir + f'/{file_name}2.png')
    # fig2.show()
    # input("Press Enter to close all figures.")
