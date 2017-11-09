from datetime import datetime, timedelta
from pprint import pprint

from rec_lib.utils import read_obj, sort_dict
import numpy as np
import matplotlib.pyplot as plt

# a = read_obj('mid_data/trainid-id-NY-Gowalla_totalCheckins/plsa1t/ex_rec-plsa1t-5.txt')


def read_checks_time(filename, split_sig='\t', uin=0, iin=4, scorein=None, timein=1, time_format='%Y-%m-%dT%H:%M:%SZ', lain=None, loin=None, toffin =None):
    table = np.zeros((7, 24))
    weekt = np.zeros(7)

    with open(filename) as f:
        for each in f:
            try:
                elements = each.strip().split(split_sig)
                # if elements[uin] != '5':
                #     continue
                _time = None if timein is None else datetime.strptime(elements[timein], time_format)
                if _time and toffin is not None:
                    offset_minutes = int(elements[toffin])
                    _time += timedelta(minutes=offset_minutes)
                table[_time.weekday(), _time.hour] += 1
                weekt[_time.weekday()] += 1
            except Exception as e:
                print(split_sig)
                raise e

    return table, weekt


def total_time_distribute():
    table ,weekt = read_checks_time('testid-id-dataset_TSMC2014_NYC.txt', uin=0,iin=1,timein=-1, time_format='%a %b %d %H:%M:%S %z %Y', toffin=-2)
    # table ,weekt = read_checks_time('testid-id-dataset_TSMC2014_TKY.txt', uin=0,iin=1,timein=-1, time_format='%a %b %d %H:%M:%S %z %Y', toffin=-2)
    # table ,weekt = read_checks_time('trainid-id-NYS-Gowalla_totalCheckins.txt', uin=0,iin=4,timein=1, time_format='%Y-%m-%dT%H:%M:%SZ')

    markers = ['o', '^', '*', '.', 'h', 'H', 'p']
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.set_xlim(xmin=0,xmax=24)
    for d in range(5):
        ax.plot(np.arange(24), table[d], marker=markers[d], label=d+1)
    ax.legend(loc=2)
    ax = fig.add_subplot(2,1,2)
    ax.set_xlim(xmin=0,xmax=24)
    for d in range(5,7):
        ax.plot(np.arange(24), table[d], marker=markers[d], label=d+1)

    ax.legend(loc=2)
    # ax = fig.add_subplot(2, 1, 2)
    # ax.plot(np.arange(7), weekt)
    # ax.set_ylim(ymin=0)

    plt.show()


def read_loc_time_distribute(filename, split_sig='\t', uin=0, iin=4, scorein=None, timein=1, time_format='%Y-%m-%dT%H:%M:%SZ', lain=None, loin=None, toffin =None):
    locTimeD = {}
    with open(filename) as f:
        for each in f:
            try:

                elements = each.strip().split(split_sig)
                # if elements[uin] != '5':
                #     continue
                loc = float(elements[iin])

                if not locTimeD.__contains__(loc):
                    locTimeD[loc] = np.zeros(24)

                _time = None if timein is None else datetime.strptime(elements[timein], time_format)
                if _time and toffin is not None:
                    offset_minutes = int(elements[toffin])
                    _time += timedelta(minutes=offset_minutes)
                locTimeD[loc][_time.hour] += 1

            except Exception as e:
                print(split_sig)
                raise e
    return sort_dict(locTimeD,key=lambda d: d[1].sum())[: 100]


def location_time_distribute():
    loc_t = read_loc_time_distribute('testid-id-dataset_TSMC2014_TKY.txt', uin=0,iin=1,timein=-1, time_format='%a %b %d %H:%M:%S %z %Y', toffin=-2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(xmin=0, xmax=24)
    for d in loc_t:
        ax.plot(np.arange(24), d[1])
    plt.show()

if __name__ == '__main__':
    # pprint(location_time_distribute())
    total_time_distribute()