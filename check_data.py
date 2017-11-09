from pprint import pprint

from numpy import array, arange
import matplotlib.pyplot as plt

from rec_lib.utils import read_obj

uinz = read_obj('mid_data/trainid-id-dataset_TSMC2014_NYC/tim-plsa7t/pr_t_in_z.txt')

for zd in uinz:
    plt.plot(arange(24), zd)

plt.show()