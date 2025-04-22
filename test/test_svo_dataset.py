from math import floor

from torch import Tensor
from torch.utils.data import DataLoader

from svo2gradslam.svo_dataset import SVOIterableDataset


def test_svo_fixture(sofa_dataset: SVOIterableDataset):
    assert len(sofa_dataset) == 3750


def test_stride(sofa_svo):
    dataset = SVOIterableDataset(str(sofa_svo), stride=2)
    assert len(dataset) == floor(3750 / 2)


def test_start_end(sofa_svo):
    assert len(SVOIterableDataset(str(sofa_svo), start=1)) == 3749
    assert len(SVOIterableDataset(str(sofa_svo), end=-1)) == 3749
    assert len(SVOIterableDataset(str(sofa_svo), start=1, end=-1)) == 3748
    assert len(SVOIterableDataset(str(sofa_svo), start=1, end=-1, stride=2)) == 3748 / 2


def test_idx_2_svo_frame_num(sofa_svo):
    dataset = SVOIterableDataset(str(sofa_svo), start=0, stride=10, end=50)
    counter = 0
    for idx, data in enumerate(dataset):
        assert dataset.idx_2_svo_frame_num(idx) == dataset.camera.get_svo_position()
        counter += 1
    assert counter == 6


def test_svo_frame_num_2_idx(sofa_svo):
    dataset = SVOIterableDataset(str(sofa_svo), start=0, stride=10, end=50)
    frames = [0, 10, 20, 30, 40, 50]
    for idx, data in enumerate(dataset):
        assert dataset.svo_frame_num_2_idx(frames[idx]) == idx


def test_svo_sofa_sofa_dataset_getitem(sofa_dataset_item):
    left_image, depth_image, intrinsics = sofa_dataset_item
    assert isinstance(left_image, Tensor)
    assert isinstance(depth_image, Tensor)
    assert isinstance(intrinsics, Tensor)
    assert intrinsics.size() == (1, 4, 4)


def test_dataloader(sofa_dataset: SVOIterableDataset):
    loader = DataLoader(sofa_dataset, batch_size=5)
    left_images, depth_images, intrinsics = next(iter(loader))
    assert isinstance(left_images, Tensor)
    assert isinstance(depth_images, Tensor)
    assert isinstance(intrinsics, Tensor)
    assert left_images.size()[0] == 5
