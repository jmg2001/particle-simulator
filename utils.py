import numpy as np
import random


def get_random_color():
    a = np.array(
        [
            random.uniform(0.5, 1.0),
            random.uniform(0.5, 1.0),
            random.uniform(0.5, 1.0),
            1.0,
        ],
        dtype=np.float32,
    )
    return a


def get_random_position(width, height):
    a = np.array(
        [
            random.randint(0, width),
            height / 2,
            0,
        ],
        dtype=np.float32,
    )
    return a


def default_position():
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def default_velocity():
    return np.array([0, 0, 0], dtype=np.float32)


def default_color():
    return get_random_color()
