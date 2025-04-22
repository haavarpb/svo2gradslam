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
