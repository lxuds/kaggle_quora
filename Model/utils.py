
"""
__file__

    utils.py

__description__

    This file provides function to convert probabilities to class.

__author__

    Lei Xu < leixuast@gmail.com >

"""

import numpy as np


## convert probability to class
def proba2class(prob):
    class_binary = np.zeros(len(prob), dtype=np.int)
    for i, p in enumerate(prob):
        if p >=0.5:
           class_binary[i] = 1
        else:
           class_binary[i] = 0
    return class_binary


