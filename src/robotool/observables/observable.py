from typing import List, Optional, Tuple

import time
import numpy as np
import trimesh
from halo import Halo
import pyrealsense2 as rs
from termcolor import colored
from src.robotool.observables.camera.multi_realsense import MultiRealsense
# from src.robotool.utils.point_cloud import create_pcd_from_rgbd_img_shape

class PointCloudAndRobotObservable:
    def __init__(
        self,
        *,
        camera_ids: List[str],
        camera_transformations: Optional[List[np.ndarray]] = None,
        camera_intrinsics: Optional[List[np.ndarray]] = None,
        height=480,
        width=640,
        pointcloud_sampled_points: int = 1024,
        pointcloud_filter_x: Tuple[float, float] = (-1, 1),
        pointcloud_filter_y: Tuple[float, float] = (-1, 1),
        pointcloud_filter_z_min: float = 0.0,
    ):
        # if camera_transformations is not None:
        #     assert len(camera_ids) == len(camera_transformations)
        # else:
        #     raise NotImplementedError("Add on-the-fly calibration")

        spinner = Halo(text='Initializing the Multi-RealSense pipeline', spinner='dots')
        spinner.start()

        ctx = rs.context()
        devices = ctx.query_devices()
        for i, dev in enumerate(devices):
            dev.hardware_reset()
            time.sleep(1)

        time.sleep(1)

        self.multi_realsense = MultiRealsense(
            serial_numbers=camera_ids,
            resolution=(width, height),
            capture_fps=30,
            put_fps=30,
            put_downsample=True,
            enable_color=True,
            enable_depth=True,
            enable_infrared=False,
            get_max_k=30,
            advanced_mode_config=None,
            transform=None,
            vis_transform=None,
            recording_transform=None,
            verbose=False
        )

        self.height = height
        self.width = width
        # self.camera_intrinsics = camera_intrinsics
        # self.camera_transformations = np.stack(camera_transformations, axis=0)
        self.pointcloud_sampled_points = pointcloud_sampled_points
        assert (
            isinstance(pointcloud_filter_x, tuple)
            and len(pointcloud_filter_x) == 2
            and pointcloud_filter_x[0] < pointcloud_filter_x[1]
        )
        assert (
            isinstance(pointcloud_filter_y, tuple)
            and len(pointcloud_filter_y) == 2
            and pointcloud_filter_y[0] < pointcloud_filter_y[1]
        )
        self.pointcloud_filter_x = pointcloud_filter_x
        self.pointcloud_filter_y = pointcloud_filter_y
        self.pointcloud_filter_z_min = pointcloud_filter_z_min
        self._gripper_site_z = None

        self.prev_obs = None
        self.started = False

        # Set exposure and white balance value
        self.set_exposure(exposureValue=150, gainValue=16)
        # self.set_white_balance(whiteBalanceValue=4800)

        spinner.stop_and_persist(symbol="✅", text="Multi-RealSense pipeline initialized")
        
    def stop(self):
        self.multi_realsense.stop()

    def start(self):
        assert not self.started, "Already started"
        self.multi_realsense.start(wait=True, put_start_time=None)
        self.started = True
        print(colored("Multi RealSense capture is ready", "green"))

    def reset(self):
        self.prev_obs = {}

    def get_obs(self, get_points=True, depth=True, infrared=True):
        obs = {}

        visual_obs = self.multi_realsense.get()
        obs["imgs"] = [item["color"] for item in visual_obs.values()]
        if depth:
            obs["depths"] = [item["depth"] for item in visual_obs.values()]
        # if infrared:
        #     obs['infrared_1'] = [item["infrared_1"] for item in visual_obs.values()]
        #     obs['infrared_2'] = [item["infrared_2"] for item in visual_obs.values()]
        if get_points:
            points = []
            for i, (rgb, depth) in enumerate(zip(obs["imgs"], obs["depths"])):
                pcd = create_pcd_from_rgbd_img_shape(depth, rgb, self.camera_intrinsics[i], self.camera_transformations[i])
                points.append(pcd)
                # print(f"Point cloud shape: {pcd.shape}")
                # print(f"rgb shape: {rgb.shape}")
                # trimesh.PointCloud(pcd.reshape(-1, 3), rgb.reshape(-1, 3)).show()
            obs["points"] = np.array(points)

        return obs

    def _get_pointcloud(self, shape, points):
        return points.reshape((shape[0], shape[1], -1)).astype(np.float32)

    def get_visual_obs(self, no_points: bool):
        """
        Get the image from the camera
        Args:
            no_points:

        Returns:

        """

        visual_obs = self.multi_realsense.get()
        imgs = [item["color"] for item in visual_obs.values()]
        depths = [item["depth"] for item in visual_obs.values()]
        timestamps = [item["timestamps"] for item in visual_obs.values()]
        # TODO: fix just getting visual observations
        return None
        # img, depth, points = imgs[0], depths[0], points[0]
        # if no_points:
        #     return {
        #         "timestamps": timestamps,
        #         "color": img,
        #         "depth": depth
        #     }
        # else:
        #     return {
        #         "timestamps": timestamps,
        #         "color": img,
        #         "depth": depth,
        #         "points": self._get_pointcloud(img, points)
        #     }
    def set_exposure(self, exposureValue, gainValue):
        self.multi_realsense.set_exposure(exposure=exposureValue, gain=gainValue)

    def set_white_balance(self, whiteBalanceValue):
        self.multi_realsense.set_white_balance(white_balance=whiteBalanceValue)