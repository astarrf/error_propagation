import numpy as np
from core import EPArray, array


def size(Earray: 'EPArray', axis=None) -> int:
    return Earray.val.size if axis is None else Earray.val.shape[axis]


def count_error(cnt_list: list):
    if all(isinstance(el, int) for el in cnt_list):
        return array(cnt_list, np.sqrt(cnt_list))
    raise ValueError("Invalid input")


def max(Earray: 'EPArray'):
    index = np.argmax(Earray.val)
    return array(Earray.val[index], Earray.sgm[index])


def min(Earray: 'EPArray'):
    index = np.argmin(Earray.val)
    return array(Earray.val[index], Earray.sgm[index])


def normalize(Earray: 'EPArray', mode='sum'):
    if mode == 'sum':
        s = np.sum(Earray.val)
        if s == 0:
            raise ValueError("Sum is zero")
        return Earray/s
    elif mode == 'max':
        return Earray/max(Earray).val
    elif mode == '01':
        maximum = max(Earray).val
        minimum = min(Earray).val
        return (Earray - minimum)/(maximum - minimum)
    elif mode == 'pm1':
        maximum = max(Earray).val
        minimum = min(Earray).val
        return 2*(Earray - minimum)/(maximum - minimum) - 1
    else:
        raise ValueError("Invalid mode")


def sum(Earray: 'EPArray', axis=None):
    return array(np.sum(Earray.val, axis), np.sqrt(np.sum(Earray.sgm**2, axis)))


def append(Earray: 'EPArray', other: 'EPArray', axis=None):
    if axis is None:
        return array(np.append(Earray.val, other.val), np.append(Earray.sgm, other.sgm))
    return array(np.append(Earray.val, other.val, axis), np.append(Earray.sgm, other.sgm, axis))
