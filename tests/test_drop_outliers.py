from module1_s4_functions import *
import pandas as pd  # type:ignore

test_array = pd.DataFrame({"a": [1, 1, 1, 1, 6, 1], "b": [3, 3, 16, 3, 3, 3]})


def test_drop_outliers_a():
    result_array = pd.DataFrame({"a": [1, 1, 1, 1, 1], "b": [3, 3, 16, 3, 3]})
    dropped = drop_outliers_by_std(test_array, columns=["a"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result_array, dropped)


def test_drop_outliers_both():
    result_array = pd.DataFrame({"a": [1, 1, 1, 1], "b": [3, 3, 3, 3]})
    dropped = drop_outliers_by_std(test_array, columns=["a", "b"]).reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(result_array, dropped)


print(drop_outliers_by_std(test_array, columns=["a", "b"]).reset_index(drop=True))
print(test_array.std())
