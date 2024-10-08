import numpy as np


def check_shape(*args):
    if not all(np.shape(args[0]) == np.shape(arg) for arg in args):
        raise ValueError("The shapes of the input are not the same")


def array(*args, rel_err_check=False):
    if len(args) == 1 and isinstance(args[0], list) and all(isinstance(el, EPArray) for el in args[0]):
        return EPArray.from_list(args[0])
    check_shape(*args)
    if len(args) == 2:
        return EPArray.from_sgm(args[0], args[1])
    elif len(args) == 3:
        if rel_err_check:
            return EPArray.from_all(args[0], args[1], args[2])
        return EPArray(args[0], args[1], args[2])
    else:
        raise ValueError("Invalid input")


def num(val: float, sgm: float, rel_err: float = None):
    if rel_err is None:
        rel_err = np.abs(sgm/val)
    return array(np.array([val]), np.array([sgm]), np.array([rel_err]), rel_err_check=False)


class EPArray:
    def __init__(self, val: np.ndarray, sgm: np.ndarray, rel_err: np.ndarray):
        # check the shape of the input
        self.val = val
        self.sgm = sgm
        self.rel_err = rel_err

    @classmethod
    def from_list(cls, error_list: list['EPArray']):
        val = np.array([e.val for e in error_list])
        sgm = np.array([e.sgm for e in error_list])
        rel_err = np.array([e.rel_err for e in error_list])
        return cls(val, sgm, rel_err)

    @classmethod
    def from_sgm(cls, val: np.ndarray, sgm: np.ndarray):
        rel_err = np.abs(sgm/val)
        return cls(val, sgm, rel_err)

    @classmethod
    def from_all(cls, val: np.ndarray, sgm: np.ndarray, rel_err: np.ndarray):
        tmp_rel_err = np.abs(sgm/val)
        if not np.allclose(tmp_rel_err, rel_err):
            raise ValueError(
                "The relative error is not consistent with the value and the error")
        return cls(val, sgm, rel_err)

    def __doable_(self, other):
        return np.isscalar(other) or np.shape(other) == np.shape(self.val) or isinstance(other, np.ndarray)

    def __add__(self, other):
        if isinstance(other, EPArray):
            return array(self.val + other.val, np.sqrt(self.sgm**2 + other.sgm**2))
        elif self.__doable_(other):
            return array(self.val + other, self.sgm)
        raise TypeError(
            "Unsupported operand type(s) for +: 'EPArray' and '{}'".format(type(other).__name__))

    def __radd__(self, other):
        if self.__doable_(other):
            return array(self.val + other, self.sgm)
        raise TypeError(
            "Unsupported operand type(s) for +: '{}' and 'EPArray'".format(type(other).__name__))

    def __sub__(self, other):
        if isinstance(other, EPArray):
            return array(self.val - other.val, np.sqrt(self.sgm**2 + other.sgm**2))
        elif self.__doable_(other):
            return array(self.val - other, self.sgm)
        raise TypeError(
            "Unsupported operand type(s) for -: 'EPArray' and '{}'".format(type(other).__name__))

    def __rsub__(self, other):
        if self.__doable_(other):
            return array(other - self.val, self.sgm)
        raise TypeError(
            "Unsupported operand type(s) for -: '{}' and 'EPArray'".format(type(other).__name__))

    def __mul__(self, other):
        if isinstance(other, EPArray):
            val = self.val * other.val
            rel_err = np.sqrt(self.rel_err**2 + other.rel_err**2)
            return array(val, np.abs(val)*rel_err, rel_err, rel_err_check=False)
        elif self.__doable_(other):
            val = self.val * other
            return array(val, np.abs(other)*self.sgm, self.rel_err, rel_err_check=False)
        raise TypeError(
            "Unsupported operand type(s) for *: 'EPArray' and '{}'".format(type(other).__name__))

    def __rmul__(self, other):
        if self.__doable_(other):
            val = other * self.val
            return array(val, np.abs(other)*self.sgm, self.rel_err, rel_err_check=False)
        raise TypeError(
            "Unsupported operand type(s) for *: '{}' and 'EPArray'".format(type(other).__name__))

    def __truediv__(self, other):
        if isinstance(other, EPArray):
            val = self.val / other.val
            rel_err = np.sqrt(self.rel_err**2 + other.rel_err**2)
            return array(val, np.abs(val)*rel_err, rel_err, rel_err_check=False)
        elif self.__doable_(other):
            if other == 0:
                raise ValueError("Division by zero")
            val = self.val / other
            return array(val, self.sgm/np.abs(other), self.rel_err, rel_err_check=False)
        raise TypeError(
            "Unsupported operand type(s) for /: 'EPArray' and '{}'".format(type(other).__name__))

    def __rtruediv__(self, other):
        if not (self.__doable_(other)):
            raise TypeError(
                "Unsupported operand type(s) for /: '{}' and 'EPArray'".format(type(other).__name__))
        if self.val.any() == 0:
            raise ValueError("Division by zero")
        val = other / self.val
        return array(val, np.abs(val)*self.rel_err, self.rel_err, rel_err_check=False)

    def __pow__(self, other):
        if isinstance(other, EPArray):
            val = self.val ** other.val
            rel_err = np.sqrt(self.rel_err**2*other.val**2 +
                              (np.log(self.val)*other.sgm)**2)
            return array(val, np.abs(val)*rel_err, rel_err, rel_err_check=False)
        elif self.__doable_(other):
            val = self.val ** other
            rel_err = np.abs(other)*self.rel_err
            return array(val, np.abs(val)*rel_err, rel_err, rel_err_check=False)
        raise TypeError(
            "Unsupported operand type(s) for **: 'EPArray' and '{}'".format(type(other).__name__))

    def __rpow__(self, other):
        if not (np.isscalar(other)):
            raise TypeError(
                "Unsupported operand type(s) for **: '{}' and 'EPArray'".format(type(other).__name__))
        val = other ** self.val
        rel_err = np.abs(np.log(other))*self.sgm
        return array(val, np.abs(val) * rel_err, rel_err, rel_err_check=False)

    def __str__(self):
        info = np.array([self.val, self.sgm]).T
        if self.val.size == 1:
            return f"[{info[0]}±{info[1]}]"
        return f"[{', '.join([f'{v[0]}±{v[1]}' for v in info])}]"

    def __repr__(self):
        return f"EPArray([{self.val}, {self.sgm}, {self.rel_err}])"
        # return self.__str__()

    def __getitem__(self, index):
        return array(self.val[index], self.sgm[index], self.rel_err[index], rel_err_check=False)

    def __setitem__(self, index, value: 'EPArray'):
        tmp_rel_err = np.abs(value.sgm/value.val)
        if not np.allclose(tmp_rel_err, value.rel_err):
            raise ValueError(
                "The relative error is not consistent with the value and the error")
        self.val[index] = value.val
        self.sgm[index] = value.sgm
        self.rel_err[index] = value.rel_err

    def __len__(self):
        return len(self.val)

    def __iter__(self):
        return iter(self.val)

    def __next__(self):
        return next(self.val)

    def __eq__(self, other: 'EPArray'):
        return np.allclose(self.val, other.val) and np.allclose(self.sgm, other.sgm) and np.allclose(self.rel_err, other.rel_err)

    def __ne__(self, other: 'EPArray'):
        return not self.__eq__(other)

    def reshape(self, *args):
        return array(self.val.reshape(*args), self.sgm.reshape(*args), self.rel_err.reshape(*args), rel_err_check=False)

    def flatten(self):
        return array(self.val.flatten(), self.sgm.flatten(), self.rel_err.flatten(), rel_err_check=False)
