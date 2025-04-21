from torch.utils.data import IterableDataset
import pyzed.sl as sl
import torch
import cv2

from importlib.resources import files


def sofa_filepath():
    return files("svo2gradslam").joinpath("sofa.svo")


class SVOIterableDataset(IterableDataset):
    def __init__(
        self,
        svo_file: str,
        sdk_verbose: int = 0,
        depth_mode: sl.DEPTH_MODE = sl.DEPTH_MODE.PERFORMANCE,
    ):
        super().__init__()

        self.svo_file = svo_file
        self.init_params = sl.InitParameters()
        self.init_params.set_from_svo_file(self.svo_file)
        self.init_params.sdk_verbose = sdk_verbose
        self.init_params.depth_mode = depth_mode
        self.camera = sl.Camera()
        self.camera.open(self.init_params)
        self.init_params = self.camera.get_init_parameters()
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.enable_depth = True
        self.sl_image = sl.Mat(mat_type=sl.MAT_TYPE.U8_C4)
        self.sl_depth = sl.Mat()

    def __len__(self):
        return self.camera.get_svo_number_of_frames()

    def __iter__(self):
        for _ in range(len(self)):

            # TODO: Let users specify the resolutions. Camera calibration must be scaled accordingly
            self.camera.grab(self.runtime_params)
            self.camera.retrieve_image(self.sl_image, sl.VIEW.LEFT)
            self.camera.retrieve_measure(self.sl_depth, measure=sl.MEASURE.DEPTH)

            # Conversion
            cv_image = cv2.cvtColor(
                self.sl_image.numpy(), cv2.COLOR_BGRA2RGB
            )  # TODO: Better to cache the destination?
            torch_image = torch.tensor(cv_image)
            image = torch_image.unsqueeze(0)

            depth_image_torch = torch.from_numpy(self.sl_depth.numpy())
            depth_image_torch = depth_image_torch.clamp(
                self.init_params.depth_minimum_distance,
                self.init_params.depth_maximum_distance,
            )
            depth_image_torch[depth_image_torch.isnan()] = 0
            depth_image = depth_image_torch.reshape((1, *depth_image_torch.size(), 1))

            intrinsics = torch.zeros((1, 4, 4))
            intrinsics[0, 0, 0] = self.get_calibration_parameters_left().fx
            intrinsics[0, 1, 1] = self.get_calibration_parameters_left().fy
            intrinsics[0, 0, 2] = self.get_calibration_parameters_left().cx
            intrinsics[0, 1, 2] = self.get_calibration_parameters_left().cy
            intrinsics[0, 2, 2] = 1
            intrinsics[0, 3, 3] = 1

            yield (image, depth_image, intrinsics)

    def get_calibration_parameters_left(self):
        return self.get_calibration_parameters().left_cam

    def get_camera_configuration(self):
        return self.camera.get_camera_information().camera_configuration

    def get_calibration_parameters(self, side: sl.SIDE = sl.SIDE.LEFT):
        return self.get_camera_configuration().calibration_parameters

    def resolution_width(self):
        return sl.get_resolution(self.init_params.camera_resolution).width

    def resolution_height(self):
        return sl.get_resolution(self.init_params.camera_resolution).height
