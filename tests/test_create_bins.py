import numpy as np
import pandas as pd  # type:ignore
from module1_s4_functions import create_bins
from typing import Union


def test_create_bins():
    # Test case 1: Create 5 bins with linear spacing
    data1 = np.linspace(1, 9, 9)
    bins1 = create_bins(data1, 3, log=False)
    assert np.array_equal(bins1, np.array([1, 5, 9]))


def test_crate_bins_log():
    # Test case 2: Create 3 bins with logarithmic spacing
    test_data = np.logspace(-1, 7, 9)
    bins = create_bins(test_data, 3, log=True)
    assert np.array_equal(bins, np.array([1e-1, 1e3, 1e7]))


test_crate_bins_log()
