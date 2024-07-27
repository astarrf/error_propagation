import numpy as np
from core import EPArray, array


def exp(Earray: 'EPArray'):
    val = np.exp(Earray.val)
    return array(val, np.abs(val*Earray.sgm))


def exp2(Earray: 'EPArray'):
    return 2**Earray


def log(Earray: 'EPArray'):
    val = np.log(Earray.val)
    return array(val, np.abs(Earray.rel_err))


def log10(Earray: 'EPArray'):
    val = np.log10(Earray.val)
    return array(val, np.abs(Earray.rel_err)/np.log(10))


def sin(Earray: 'EPArray'):
    val = np.sin(Earray.val)
    return array(val, np.abs(np.cos(Earray.val)*Earray.sgm))


def cos(Earray: 'EPArray'):
    val = np.cos(Earray.val)
    return array(val, np.abs(np.sin(Earray.val)*Earray.sgm))


def tan(Earray: 'EPArray'):
    val = np.tan(Earray.val)
    return array(val, np.abs(1/np.cos(Earray.val)**2*Earray.sgm))


def arcsin(Earray: 'EPArray'):
    val = np.arcsin(Earray.val)
    return array(val, np.abs(1/np.sqrt(1 - Earray.val**2)*Earray.sgm))


def arccos(Earray: 'EPArray'):
    val = np.arccos(Earray.val)
    return array(val, np.abs(1/np.sqrt(1 - Earray.val**2)*Earray.sgm))


def arctan(Earray: 'EPArray'):
    val = np.arctan(Earray.val)
    return array(val, np.abs(1/(1 + Earray.val**2)*Earray.sgm))


def sinh(Earray: 'EPArray'):
    val = np.sinh(Earray.val)
    return array(val, np.abs(np.cosh(Earray.val)*Earray.sgm))


def cosh(Earray: 'EPArray'):
    val = np.cosh(Earray.val)
    return array(val, np.abs(np.sinh(Earray.val)*Earray.sgm))


def tanh(Earray: 'EPArray'):
    val = np.tanh(Earray.val)
    return array(val, np.abs(1/np.cosh(Earray.val)**2*Earray.sgm))


def sqrt(Earray: 'EPArray'):
    return Earray**0.5


def abs(Earray: 'EPArray'):
    return array(np.abs(Earray.val), Earray.sgm)
