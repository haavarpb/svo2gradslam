from torch import Tensor


def test_fixtures(sofa_dataset_item):
    color, depth, intrinsics = sofa_dataset_item
    assert isinstance(color, Tensor)
