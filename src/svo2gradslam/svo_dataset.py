from importlib.resources import files
from math import floor

import cv2
import pyzed.sl as sl
import torch
from torch.utils.data import IterableDataset


def sofa_filepath():
    return files("svo2gradslam").joinpath("sofa.svo")


class SVOIterableDataset(IterableDataset):
    def __init__(
        self,
        svo_file: str,
        sdk_verbose: int = 0,
        depth_mode: sl.DEPTH_MODE = sl.DEPTH_MODE.PERFORMANCE,
        start=0,
        end=None,
        stride=1,
        **kwargs,
    ):
        super().__init__()

        self.desired_height = 0 if "desired_height" not in kwargs.keys() else kwargs["desired_height"]
        self.desired_width = 0 if "desired_width" not in kwargs.keys() else kwargs["desired_width"]
        self.resolution_resizer = sl.Resolution(self.desired_width, self.desired_height)
        self.svo_file = svo_file
        self.init_params = sl.InitParameters()
        self.init_params.set_from_svo_file(self.svo_file)
        self.init_params.sdk_verbose = sdk_verbose # Check how to shut up zed camera print logging
        self.init_params.depth_mode = depth_mode
        self.camera = sl.Camera()
        self.camera.open(self.init_params) # TODO: Defer opening
        self.init_params = self.camera.get_init_parameters()
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.enable_depth = True
        self.sl_image = sl.Mat(mat_type=sl.MAT_TYPE.U8_C4)
        self.sl_depth = sl.Mat()
        self.stride = stride
        self.start = start
        self.end = end


    def __len__(self):
        if self.end is None:  # End is very last svo frame
            self.end = self.camera.get_svo_number_of_frames()
        elif self.end < 0:  # End specifies how many trailing frames to slice
            self.end = self.camera.get_svo_number_of_frames() - abs(self.end)
        elif self.end > 0:  # End specifies last frame
            assert self.end <= self.camera.get_svo_number_of_frames()
        return floor((self.end - self.start) / self.stride)

    def idx_2_svo_frame_num(self, idx: int):
        return self.start + self.stride * idx

    def svo_frame_num_2_idx(self, frame_num: int):
        return floor((frame_num - self.start) / self.stride)

    def __iter__(self):
        self.camera.set_svo_position(self.start)
        for idx in range(len(self) + 1):
            if self.stride != 1:
                self.camera.set_svo_position(self.idx_2_svo_frame_num(idx))
            # TODO: Let users specify the resolutions. Camera calibration must be scaled accordingly
            image, depth_image, intrinsics = self.get_frame()

            yield (image, depth_image, intrinsics)

    def get_frame(self):
        self.camera.grab(self.runtime_params)
        self.camera.retrieve_image(self.sl_image, sl.VIEW.LEFT, resolution=self.resolution_resizer)
        self.camera.retrieve_measure(self.sl_depth, measure=sl.MEASURE.DEPTH, resolution=self.resolution_resizer)

            # Conversion
        cv_image = cv2.cvtColor(
                self.sl_image.numpy(), cv2.COLOR_BGRA2RGB
            )  # TODO: Better to cache the destination?
        image = torch.tensor(cv_image, dtype=torch.float32)

        depth_image_torch = torch.from_numpy(self.sl_depth.numpy())
        depth_image_torch = depth_image_torch.clamp(
                self.init_params.depth_minimum_distance,
                self.init_params.depth_maximum_distance,
            )
        depth_image_torch[depth_image_torch.isnan()] = 0
        depth_image = depth_image_torch.reshape((*depth_image_torch.size(), 1))

        intrinsics = torch.zeros((4, 4))
        intrinsics[0, 0] = self.get_calibration_parameters_left().fx
        intrinsics[1, 1] = self.get_calibration_parameters_left().fy
        intrinsics[0, 2] = self.get_calibration_parameters_left().cx
        intrinsics[1, 2] = self.get_calibration_parameters_left().cy
        intrinsics[2, 2] = 1
        intrinsics[3, 3] = 1
        return image, depth_image, intrinsics

    def __getitem__(self, idx):
        self.camera.set_svo_position(self.idx_2_svo_frame_num(idx))
        return self.get_frame()


    def get_calibration_parameters_left(self):
        return self.get_calibration_parameters().left_cam

    def get_camera_configuration(self):
        return self.camera.get_camera_information(resizer=self.resolution_resizer).camera_configuration

    def get_calibration_parameters(self, side: sl.SIDE = sl.SIDE.LEFT):
        return self.get_camera_configuration().calibration_parameters

    def resolution_width(self):
        return sl.get_resolution(self.init_params.camera_resolution).width

    def resolution_height(self):
        return sl.get_resolution(self.init_params.camera_resolution).height
    
    def get_resolution(self):
        return (self.resolution_height(), self.resolution_width())
    



from torch import Tensor
from torch.utils.data import default_collate


def collate_sequence(batch : list[tuple[Tensor, Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor]:
    c,d,i = default_collate(batch)
    c = c.unsqueeze(0)
    d = d.unsqueeze(0)
    i = i[0:1].unsqueeze(0)
    return (c,d,i)
