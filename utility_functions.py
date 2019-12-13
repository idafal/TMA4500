import numpy as np
from itertools import groupby


def get_n_in_row(array, n):
    """
    Find the first cascade
    :param array: binary array
    :param n: number of subsequent equal elements
    :return: value of the first cascade, false if there is no cascade (n subsequent equal decisions)
    """
    for key, group in groupby(array):
        i = sum(1 for _ in group)
        if i>=n:
            return int(key)
    return -1


def get_groups(array):
    """
    Return dictionary with number of preceding equal numbers in array, with number as key
    """
    groups = []
    for key, group in groupby(array):
        i = sum(1 for _ in group)
        groups.append((int(key), i))
    return groups

def change_in_cascade(groups, n):
    number_of_changes = 0
    wrong_changes = 0 # count number of time a cascade changes from correct to wrong by one or more individual
    one_change = 0 # count if the cascade changes once by at least one individual
    new_cascades = 0 # count number of times a new cascade occurs (at least n in a row) (if less than 10 deviates within one cascade, it is counted as one)
    singles = 0

    if len(groups) == 1:
        return -1

    indices = np.asarray(groups)[:, 0]
    values = np.asarray(groups)[:, 1]

    for i in range(len(groups)):
        if groups[i][1] >= n:
            number_of_changes = len(groups[i + 1:])
            one_change = 1  * (len(groups[i:]) > 1)
            wrong_changes = np.sum(indices[i:]) - (1 * int(indices[i] == 1)) # dont count current wrong
            new_cascades = np.sum(values[i + 1:] >= n)
            singles = np.sum(values[i+1:]<n)

            return number_of_changes, one_change, wrong_changes, new_cascades, singles
    return number_of_changes, one_change, wrong_changes, new_cascades, singles