import numpy as np


def array(*args):
    if len(args) == 1 and isinstance(args[0], list) and all(isinstance(el, EPArray) for el in args[0]):
        return EPArray.from_list(args[0])
    elif len(args) == 2:
        return EPArray(args[0], args[1])
    else:
        raise ValueError("Invalid input")


class EPArray:
    def __init__(self, val: np.ndarray, sgm: np.ndarray, rel_err: np.ndarray = None):
        # check the shape of the input
        if np.shape(val) != np.shape(sgm):
            raise ValueError(
                "The shape of the value and the error must be the same")
        if rel_err is not None and np.shape(val) != np.shape(rel_err):
            raise ValueError(
                "The shape of the value and the relative error must be the same")
        self.val = np.array(val)
        self.sgm = np.array(sgm)
        self.rel_err = np.abs(
            self.sgm/self.val) if rel_err is None else rel_err

    @classmethod
    def from_list(cls, error_list: list['EPArray']):
        val = np.array([e.val for e in error_list])
        sgm = np.array([e.sgm for e in error_list])
        return cls(val, sgm)

    def __doable__(self, other):
        return np.isscalar(other) or np.shape(other) == np.shape(self.val) or isinstance(other, np.ndarray)

    def __add__(self, other):
        if isinstance(other, EPArray):
            return EPArray(self.val + other.val, np.sqrt(self.sgm**2 + other.sgm**2))
        elif self.__doable__(other):
            return EPArray(self.val + other, self.sgm)
        raise TypeError(
            "Unsupported operand type(s) for +: 'EParray' and '{}'".format(type(other).__name__))

    def __radd__(self, other):
        if self.__doable__(other):
            return EPArray(self.val + other, self.sgm)
        raise TypeError(
            "Unsupported operand type(s) for +: '{}' and 'EParray'".format(type(other).__name__))

    def __sub__(self, other):
        if isinstance(other, EPArray):
            return array(self.val - other.val, np.sqrt(self.sgm**2 + other.sgm**2))
        elif self.__doable__(other):
            return array(self.val - other, self.sgm)
        raise TypeError(
            "Unsupported operand type(s) for -: 'EParray' and '{}'".format(type(other).__name__))

    def __rsub__(self, other):
        if self.__doable__(other):
            return array(other - self.val, self.sgm)
        raise TypeError(
            "Unsupported operand type(s) for -: '{}' and 'EParray'".format(type(other).__name__))

    def __mul__(self, other):
        if isinstance(other, EPArray):
            val = self.val * other.val
            return array(val, np.abs(val)*np.sqrt(self.rel_err**2 + other.rel_err**2))
        elif self.__doable__(other):
            val = self.val * other
            return array(val, np.abs(other)*self.sgm)
        raise TypeError(
            "Unsupported operand type(s) for *: 'EParray' and '{}'".format(type(other).__name__))

    def __rmul__(self, other):
        if self.__doable__(other):
            val = other * self.val
            return array(val, np.abs(other)*self.sgm)
        raise TypeError(
            "Unsupported operand type(s) for *: '{}' and 'EParray'".format(type(other).__name__))

    def __truediv__(self, other):
        if isinstance(other, EPArray):
            val = self.val / other.val
            return array(val, np.abs(val)*np.sqrt(self.rel_err**2 + other.rel_err**2))
        elif self.__doable__(other):
            if other == 0:
                raise ValueError("Division by zero")
            val = self.val / other
            return array(val, self.sgm/np.abs(other))
        raise TypeError(
            "Unsupported operand type(s) for /: 'EParray' and '{}'".format(type(other).__name__))

    def __rtruediv__(self, other):
        if not (self.__doable__(other)):
            raise TypeError(
                "Unsupported operand type(s) for /: '{}' and 'EParray'".format(type(other).__name__))
        if self.val.any() == 0:
            raise ValueError("Division by zero")
        val = other / self.val
        return array(val, np.abs(val)*self.rel_err)

    def __pow__(self, other):
        if isinstance(other, EPArray):
            val = self.val ** other.val
            return array(val, np.abs(val)*np.sqrt(self.rel_err**2*other.val**2 + (np.log(self.val)*other.sgm)**2))
        elif self.__doable__(other):
            val = self.val ** other
            return array(val, np.abs(val*other)*self.rel_err)
        raise TypeError(
            "Unsupported operand type(s) for **: 'EParray' and '{}'".format(type(other).__name__))

    def __rpow__(self, other):
        if not (np.isscalar(other)):
            raise TypeError(
                "Unsupported operand type(s) for **: '{}' and 'EParray'".format(type(other).__name__))
        val = other ** self.val
        return array(val, np.abs(val*np.log(other))*self.sgm)

    def __str__(self):
        return f"[{self.val}, {self.sgm}]"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, index):
        return array(self.val[index], self.sgm[index])

    def __setitem__(self, index, value):
        self.val[index] = value.val
        self.sgm[index] = value.sgm
