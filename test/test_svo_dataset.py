from svo2gradslam.svo_dataset import SVOIterableDataset
from pathlib import Path
from torch import Tensor
import pytest
from importlib.resources import files
from torch.utils.data import DataLoader


@pytest.fixture
def sofa_svo():
    return files("svo2gradslam").joinpath("sofa.svo")


@pytest.fixture
def sofa_dataset(sofa_svo):
    return SVOIterableDataset(str(sofa_svo))


@pytest.fixture
def sofa_dataset_item(sofa_dataset):
    return next(iter(sofa_dataset))


def test_svo_fixture(sofa_dataset):
    assert len(sofa_dataset) == 3750


def test_svo_sofa_sofa_dataset_getitem(sofa_dataset_item):
    left_image, depth_image, intrinsics = sofa_dataset_item
    assert isinstance(left_image, Tensor)
    assert isinstance(depth_image, Tensor)
    assert isinstance(intrinsics, Tensor)
    assert intrinsics.size() == (1, 4, 4)


def test_dataloader(sofa_dataset):
    loader = DataLoader(sofa_dataset, batch_size=5)
    left_images, depth_images, intrinsics = next(iter(loader))
    assert isinstance(left_images, Tensor)
    assert isinstance(depth_images, Tensor)
    assert isinstance(intrinsics, Tensor)
    assert left_images.size()[0] == 5
