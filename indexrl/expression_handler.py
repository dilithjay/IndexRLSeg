import numpy as np
import random


def eval_expression(exp: list, image: np.ndarray = None):
    expression = ""

    for token in exp:
        if token[0] == "c":
            channel = eval(token[1:])
            expression += f"(image[{channel}] + {(random.random() * 0.001 + 1) * 1e-5})"  # To prevent divide by zero
        elif token == "sq":
            expression += "**2"
        elif token == "sqrt":
            expression += "**0.5"
        elif token == "=":
            break
        else:
            expression += token

    try:
        return eval(expression)
    except (SyntaxError, FloatingPointError):
        return False
    except TypeError:
        return False


class UnitNode(object):
    def __init__(self, level=0):
        self.level = level

    def __add__(self, obj):
        if obj.level == self.level:
            return UnitNode(self.level)
        raise ValueError

    def __sub__(self, obj):
        return self + obj

    def __truediv__(self, obj):
        return UnitNode(self.level - obj.level)

    def __mul__(self, obj):
        return UnitNode(self.level + obj.level)

    def __pow__(self, val):
        return UnitNode(self.level * val)


def check_unitless_validity(exp: list):
    expression = ""
    for token in exp:
        if token[0] == "c":
            expression += "UnitNode(1)"
        elif token == "sq":
            expression += "**2"
        elif token == "sqrt":
            expression += "**0.5"
        elif token == "-":
            expression += "+"
        elif token == "=":
            break
        else:
            expression += token
    try:
        res = eval(expression)
        if res.level == 0:
            return True
        return res.level
    except ValueError:
        return 1
    except (SyntaxError, FloatingPointError):
        return False
