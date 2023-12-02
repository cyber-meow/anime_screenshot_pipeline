import pytest
import numpy as np
from anime2sd.classif import select_indices_recursively


def test_even_distribution():
    list_of_indices = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    n_to_select = 6
    expected_result = [1, 2, 4, 5, 7, 8]
    assert select_indices_recursively(list_of_indices, n_to_select) == expected_result


def test_uneven_distribution():
    list_of_indices = [np.array([1, 2]), np.array([3, 4, 5, 6]), np.array([7])]
    n_to_select = 5
    expected_result = [1, 2, 3, 4, 7]
    assert (
        sorted(select_indices_recursively(list_of_indices, n_to_select))
        == expected_result
    )


def test_more_to_select_than_available():
    list_of_indices = [np.array([1]), np.array([2, 3]), np.array([4, 5])]
    n_to_select = 10
    expected_result = [1, 2, 3, 4, 5]
    assert (
        sorted(select_indices_recursively(list_of_indices, n_to_select))
        == expected_result
    )


def test_empty_arrays():
    list_of_indices = [np.array([]), np.array([])]
    n_to_select = 5
    expected_result = []
    assert select_indices_recursively(list_of_indices, n_to_select) == expected_result


def test_no_arrays():
    list_of_indices = []
    n_to_select = 5
    expected_result = []
    assert select_indices_recursively(list_of_indices, n_to_select) == expected_result


def test_zero_to_select():
    list_of_indices = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    n_to_select = 0
    expected_result = []
    assert select_indices_recursively(list_of_indices, n_to_select) == expected_result


# Add more tests if necessary to cover additional scenarios or edge cases.

# Run the tests
if __name__ == "__main__":
    pytest.main()
