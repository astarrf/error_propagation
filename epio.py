import numpy as np
from .core import EPArray, array


def save(path, e_array: 'EPArray') -> None:
    np.save(path, [e_array.val, e_array.sgm, e_array.rel_err])


def load(path):
    val, sgm, rel_err = np.load(path, allow_pickle=True)
    e_array = array(val, sgm, rel_err)
    return e_array
