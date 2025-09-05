import argparse
import yaml
import os.path
import time
import math
import cv2
import numpy as np
import numpy.matlib as npm
import pupil_apriltags as apriltag
import pyrealsense2 as rs


EPS = np.finfo(float).eps * 4.0
# DIST_FROM_ORIGIN = 0.03 * 1.37777777
DIST_FROM_ORIGIN = 6.33 / 100

DEBUG = True

def show_detection(img, detection):
    """
    open cv2 window
    show image with detection overlay
    hint: detection.corners are the pixel coordinates of the four tag corners
    """
    corners = detection.corners.astype(int)
    cv2.polylines(img, [corners], True, (0, 0, 255), 2)
    cv2.imshow("apriltags", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rot_mat(angles, hom: bool = False):
    """Given @angles (x, y, z), compute rotation matrix
    Args:
        angles: (x, y, z) rotation angles.
        hom: whether to return a homogeneous matrix.
    """
    x, y, z = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    if hom:
        M = np.zeros((4, 4), dtype=np.float32)
        M[:3, :3] = R
        M[3, 3] = 1.0
        return M
    return R


def get_mat(pos, angles):
    """Get homogeneous matrix given position and rotation angles.
    Args:
        pos: relative positions (x, y, z).
        angles: relative rotations (x, y, z) or 3x3 matrix.
    """
    transform = np.zeros((4, 4), dtype=np.float32)
    if not isinstance(angles, np.ndarray) or not len(angles.shape) == 2:
        transform[:3, :3] = np.eye(3) if not np.any(angles) else rot_mat(angles)
    else:
        if len(angles[0, :]) == 4:
            transform[:4, :4] = angles
        else:
            transform[:3, :3] = angles
    transform[3, 3] = 1.0
    transform[:3, 3] = pos
    return transform


def to_homogeneous(pos, rot):
    """Givien position and rotation matrix, convert it into homogeneous matrix."""
    if isinstance(pos, list):
        pos = np.array(pos)
    transform = np.zeros((4, 4))
    if pos.ndim == 2:
        transform[:3, 3:] = pos
    else:
        assert pos.ndim == 1
        transform[:3, 3] = pos
    transform[:3, :3] = rot
    transform[3, 3] = 1

    return transform


def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.
    Args:
        rmat: 3x3 rotation matrix
    Returns:
        vec4 float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


def convert_quat(q, to="xyzw"):
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q: a 4-dim numpy array corresponding to a quaternion
        to: a string, either 'xyzw' or 'wxyz', determining
            which convention to convert to.
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


def quat2mat(quaternion):
    """
    Converts given quaternion (x, y, z, w) to matrix.
    Args:
        quaternion: vec4 float angles
    Returns:
        3x3 rotation matrix
    """
    # awkward semantics for use with numba

    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3).astype(np.float32)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ],
        dtype=np.float32,
    )


def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4, 4))

    for i in range(0, M):
        q = Q[i, :]
        # multiply q with its transposed version q' and add A
        A = np.outer(q, q) + A

    # scale
    A = (1.0 / M) * A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:, 0].A1)


def comp_avg_pose(poses):
    np.set_printoptions(suppress=True)
    quats = []
    positions = []
    for pose in poses:
        if pose is None:
            continue
        quats.append(convert_quat(mat2quat(pose[:3, :3]), "wxyz"))
        positions.append(pose[:3, 3])

    quats = np.stack(quats, axis=0)
    positions = np.stack(positions, axis=0)

    avg_quat = averageQuaternions(quats).astype(np.float32)

    avg_rot = quat2mat(convert_quat(avg_quat, "xyzw"))
    avg_pos = positions.mean(axis=0)
    return to_homogeneous(avg_pos, avg_rot)


def get_cam_to_base(cam, april_tag):
    """Get homogeneous transforms that maps camera points to base points."""
    color_frame, _ = cam.get_frame()
    intr = cam.color_intr_param
    color_frame = np.asanyarray(color_frame.get_data())
    color_frame = cv2.undistort(
        color_frame, cam.color_intrinsic_matrix, cam.color_distortion, None
    )

    tags = april_tag.detect_id(color_frame, intr)

    trials = 10
    cam_to_bases = []
    for _ in range(trials):
        for base_tag in tags.values():
            rel_pose = REL_POSE_FROM_APRIL_TAG_COORDINATE_ORIGIN[base_tag.tag_id]
            transform = to_homogeneous(base_tag.pose_t, base_tag.pose_R) @ np.linalg.inv(rel_pose)
            cam_to_bases.append(np.linalg.inv(transform))

        tags = april_tag.detect_id(color_frame, intr)
        time.sleep(0.01)

    for base_tag in tags.values():
        show_detection(color_frame, base_tag)

    #  All the base tags are not detected.
    if len(cam_to_bases) == 0:
        raise Exception("Base tags are not detected.")
    cam_to_base = comp_avg_pose(cam_to_bases)
    if DEBUG:
        print("Average pose: ")
        print(cam_to_base)
    return cam_to_base


REL_POSE_FROM_APRIL_TAG_COORDINATE_ORIGIN = {
    0: get_mat([-DIST_FROM_ORIGIN, -DIST_FROM_ORIGIN, 0], [0, 0, 0]),
    1: get_mat([DIST_FROM_ORIGIN, -DIST_FROM_ORIGIN, 0], [0, 0, 0]),
    2: get_mat([-DIST_FROM_ORIGIN, DIST_FROM_ORIGIN, 0], [0, 0, 0]),
    3: get_mat([DIST_FROM_ORIGIN, DIST_FROM_ORIGIN, 0], [0, 0, 0]),
}


class AprilTag:
    def __init__(self, tag_size):
        self.at_detector = apriltag.Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            debug=0,
        )
        self.tag_size = tag_size

    def detect(self, frame, intr_param):
        """Detect AprilTag.

        Args:camera_ext_calibration.yaml
            frame: pyrealsense2.frame or Gray-scale image to detect AprilTag.
            intr_param: Camera intrinsics format of [fx, fy, ppx, ppy].
        Returns:
            Detected tags.
        """
        if isinstance(frame, rs.frame):
            frame = np.asanyarray(frame.get_data())
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        detections = self.at_detector.detect(frame, True, intr_param, self.tag_size)
        # Filter out bad detections.
        detections = sorted(detections, key=lambda x: x.tag_id)
        return [detection for detection in detections if detection.hamming < 2]

    def detect_id(self, frame, intr_param):
        detections = self.detect(frame, intr_param)
        # Make it as a dictionary which the keys are tag_id.
        return {detection.tag_id: detection for detection in detections}


class RealsenseCam:
    def __init__(
        self,
        serial,
        color_res,
        depth_res,
        frame_rate,
        disable_auto_exposure: bool = False,
    ):
        self.started = False
        self.serial = serial

        self.color_res = color_res
        self.depth_res = depth_res
        self.frame_rate = frame_rate
        # Create a context object. This object owns the handles to all connected realsense devices
        self.pipeline = rs.pipeline()
        # Configure streams
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, *self.color_res, rs.format.rgb8, self.frame_rate)
        config.enable_stream(rs.stream.depth, *self.depth_res, rs.format.z16, self.frame_rate)
        # Start streaming
        try:
            conf = self.pipeline.start(config)
        except Exception as e:
            print(f"[Error] Could not initialize camera serial: {self.serial}")
            raise e

        self.min_depth = 0.15
        self.max_depth = 2.0
        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.min_distance, self.min_depth)
        self.threshold_filter.set_option(rs.option.max_distance, self.max_depth)

        # Get intrinsic parameters of color image
        color_profile = conf.get_stream(rs.stream.color)
        self.color_intrinsic = (
            color_intr_param
        ) = (
            color_profile.as_video_stream_profile().get_intrinsics()
        )  # Downcast to video_stream_profile and fetch intrinsics
        self.color_intr_param = [
            color_intr_param.fx,
            color_intr_param.fy,
            color_intr_param.ppx,
            color_intr_param.ppy,
        ]

        # Get intrinsic parameters of depth image
        depth_profile = conf.get_stream(rs.stream.depth)
        self.depth_intrinsic = depth_intr_param = depth_profile.as_video_stream_profile().get_intrinsics()
        self.depth_intr_param = [
            depth_intr_param.fx,
            depth_intr_param.fy,
            depth_intr_param.ppx,
            depth_intr_param.ppy,
        ]

        # Get the sensor once at the beginning. (Sensor index: 1)
        color_sensor = conf.get_device().first_color_sensor()
        # Set the exposure anytime during the operation
        color_sensor.set_option(rs.option.enable_auto_exposure, True)

        # Set region of interest.
        # color_sensor = conf.get_device().first_roi_sensor()

        if disable_auto_exposure:
            color_sensor.set_option(rs.option.enable_auto_exposure, False)

        for _ in range(10):
            # Read dummy observation to setup exposure.
            self.get_frame()
            time.sleep(0.04)

        self.started = True
        self.color_distortion = np.array(color_intr_param.coeffs)


    def get_frame(self):
        """Read frame from the realsense camera.

        Returns:
            Tuple of color and depth image. Return None if failed to read frame.

            color frame:(height, width, 3) RGB uint8 realsense2.video_frame.
            depth frame:(height, width) z16 realsense2.depth_frame.
        """
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        depth_frame = self.threshold_filter.process(depth_frame)

        if not color_frame or not depth_frame:
            return None, None
        return color_frame, depth_frame

    def get_image(self):
        """Get numpy color and depth image.

        Returns:
            Tuble of numpy color and depth image. Return (None, None) if failed.

            color image: (height, width, 3) RGB uint8 numpy array.
            depth image: (height, width) z16 numpy array.
        """
        color_frame, depth_frame = self.get_frame()
        if color_frame is None or depth_frame is None:
            return None, None
        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data()).copy()
        depth_image = np.asanyarray(depth_frame.get_data()).copy()

        return color_image, depth_image

    def __del__(self):
        if self.started:
            self.pipeline.stop()

    @property
    def color_intrinsic_matrix(self):
        return np.array(
            [
                [self.color_intrinsic.fx, 0.0, self.color_intrinsic.ppx],
                [0.0, self.color_intrinsic.fy, self.color_intrinsic.ppy],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    def depth_intrinsic_matrix(self):
        return np.array(
            [
                [self.depth_intrinsic.fx, 0.0, self.depth_intrinsic.ppx],
                [0.0, self.depth_intrinsic.fy, self.depth_intrinsic.ppy],
                [0.0, 0.0, 1.0],
            ]
        )


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--serial_numbers", nargs="+", help="<Required> all serial numbers", required=False)
    args.add_argument("--output_yaml", type=str, default="camera_ext_calibration_0901.yaml")
    args.add_argument("--gripper_cam_serial", type=str, default="")
    args = args.parse_args()

    # check if there exists an yaml extrinsic file already
    if os.path.exists(args.output_yaml):
        overwrite = input(f"File {args.output_yaml} already exists. Overwrite? (y/n): ").lower()
        if overwrite != "y":
            return
        os.remove(args.output_yaml)

    output_dict = []
    camera_ids = ["244222072252", 
                     "250122071059",
                     "125322060645",
                     "246422071990",
                    # "123122060454",
                     "246422071818",
                     "246422070730",
                     "250222072777",
                     "204222063088"]
    for id, serial in enumerate(camera_ids):

        print(f"Calibrating camera {id} with serial number {serial}")

        cam = RealsenseCam(
            serial,
            (640, 480),
            (640, 480),
            30,
            disable_auto_exposure=True,
        )
        # april_tag = AprilTag(0.045*1.37777777)
        april_tag = AprilTag(0.05)
        cam_to_april = get_cam_to_base(cam, april_tag)
        # this is the transformation from april tag to Franka base, can be measured on the workspace
        april_to_base = get_mat([0, 0, 0], [np.pi, 0, np.pi / 2])
        camera_to_base = april_to_base @ cam_to_april

        print("Camera to base transformation:\n", camera_to_base)

        output_dict.append(
            {
                "camera_id": id,
                "serial_number": serial,
                "transformation": camera_to_base.tolist(),
                "color_intrinsic_matrix": cam.color_intrinsic_matrix.tolist(),
                "depth_intrinsic_matrix": cam.depth_intrinsic_matrix.tolist(),
            }
        )

    with open(args.output_yaml, "w") as f:
        yaml.dump(output_dict, f, default_flow_style=False)

    print("success!")


if __name__ == "__main__":
    main()


# gripper cam serial number: 821312060498