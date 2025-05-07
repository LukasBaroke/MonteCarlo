import numpy as np
from montecarlo.utils import compute_observable
from montecarlo.bitstring import BitString


def test_compute_observable():
    config = np.array([1, -1, 1, -1])
    assert compute_observable(config) == 0
