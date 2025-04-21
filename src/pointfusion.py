from argparse import ArgumentParser

import open3d as o3d
import torch
from torch.utils.data import DataLoader

from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages
from svo2gradslam.svo_dataset import SVOIterableDataset, sofa_filepath

parser = ArgumentParser()
parser.add_argument(
    "--svo-file", required=False, default=str(sofa_filepath()), dest="svo_file"
)
args = parser.parse_args()

if __name__ == "__main__":
    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = SVOIterableDataset(args.svo_file)
    # load dataset
    loader = DataLoader(dataset=dataset, batch_size=2)
    colors, depths, intrinsics = next(iter(loader))

    # create rgbdimages object
    rgbdimages = RGBDImages(colors, depths, intrinsics)

    # SLAM
    slam = PointFusion(dsratio=4, device=device)
    pointclouds, recovered_poses = slam(rgbdimages)

    # visualization
    o3d.visualization.draw_geometries([pointclouds.open3d(0)])
