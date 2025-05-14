from pathlib import Path

import numpy as np
import pytest
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


@pytest.fixture
def mask_generator():
    sam = sam_model_registry['vit_b'](checkpoint=Path("~/phd/sam_vit_b_01ec64.pth").expanduser())
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )
    return mask_generator


def test_load_numpyarray(mask_generator):
    arr = np.load("/home/havarbra/svo2gradslam/test/image_that_generated_float_bbox.npy")
    masks = mask_generator.generate(arr)

    if masks:
        assert any(map(lambda x: x['bbox'][0] is int, masks))