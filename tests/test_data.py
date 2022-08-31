import numpy as np

from smallteacher.data import ListOfDatasetsWithMosaicing


def test_combined_dataset_without_list():

    a = ListOfDatasetsWithMosaicing(np.array([1, 1, 1]))
    assert len(a) == 3
    assert a[0] == 1


def test_combined_dataset_with_list():
    a = ListOfDatasetsWithMosaicing([np.array([1, 1, 1]), np.array([2, 2, 2])])
    assert len(a) == 6
    for i in range(len(a)):
        if i <= 2:
            assert a[i] == 1
        else:
            assert a[i] == 2
