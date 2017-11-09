# -*- coding:utf-8 -*-
import json

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def threeDemsion():
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    # 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

    plt.show()


# data：每条线的数据，
# line_names：每条线的名字
# xlias：x坐标别名
def line(data, line_names, xlias, ymax=0.1, ymin=0, title='default title', xlabel='x', ylabel='y'):
    import matplotlib.pyplot as plt
    markers = ['o', '*', 'D', 'd', 's', 'p', '^', 'H', 'h']
    plt.title(title)
    plt.ylim(ymax=ymax, ymin=ymin)
    x_data = range(len(xlias))
    for i in range(len(data)):
        plt.plot(x_data, data[i], marker=markers[i], label=line_names[i])
    plt.xticks(x_data, xlias)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


# data：每条线的数据，
# labels：每组图的名字
# xlias：x坐标别名
def bar(data, labels, xlias, ymax, ymin, xlabel, ylabel, title):
    import matplotlib as mpl
    import numpy as np
    import matplotlib.pyplot as plt
    n_groups = len(data[0])
    colors = ['r', 'g', 'c', 'm', 'w', 'b', 'y']
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.3
    opacity = 0.5
    for n in range(len(data)):
        plt.bar(index + bar_width / 2 * n, data[n], bar_width / 2, alpha=opacity, color=colors[n], label=labels[n])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(index - opacity / 2 + 2 * bar_width, xlias, fontsize=18)

    plt.yticks(fontsize=18)

    plt.ylim(ymin, ymax)
    plt.legend(loc='upper ')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    precison = np.array([
        [0.112, 0.102, 0.122, 0.106, 0.120, 0.131, 0.125, 0.112],
        [0.068, 0.065, 0.062, 0.073, 0.067, 0.0701, 0.0710, 0.065]
    ])
    precison[0] *= 0.32
    precison[1] *= 0.44

    line(precison,
         ['tokyo', 'newyork'],
         [3, 4, 5, 6, 7, 8, 9, 10],
         ymax=0.05,
         title='recall@10',
         xlabel='topic-nums',
         ylabel='precision')
