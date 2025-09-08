import argparse
import os
import time
import rospy
from std_msgs.msg import Float32MultiArray, Int32, String
import numpy as np
from tele_vision import OpenTeleVision
import ros_numpy
import cv2
import threading
from camera_utils import list_video_devices, find_device_path_by_name
from multiprocessing import shared_memory
import sys
import signal
import h5py
from datetime import datetime
import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import pupil_apriltags as apriltag
import yaml
import json
import uuid

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class HumanDataCollector:
    def __init__(self, args):
        rospy.init_node('avp_teleoperation')
        self.args = args
        self.checkpt = 0
        self.manipulate_eef_idx = [0]
        self.update_manipulate_eef_idx()
        
        # Initialize ROS publisher for collection status
        self.collection_status_pub = rospy.Publisher('/avp/collection_status', String, queue_size=10)
        
        # Initialize ROS subscriber for acknowledgment from other scripts
        self.ack_subscriber = rospy.Subscriber('/avp/acknowledgment', String, self.ack_callback)
        self.other_script_ready = False
        self.waiting_for_ack = False
        self.ack_wait_start_time = None
        self.ack_timeout = 45.0  # 45 seconds timeout
        self.collection_acknowledged = False  # New state to track when collection can proceed


        self.head_intr_param = None  # [fx, fy, ppx, ppy] for realsense head color camera
        self.head_intr_param_left = None  # stereo left intrinsics
        self.head_intr_param_right = None  # stereo right intrinsics
        self.head_dist_left = None
        self.head_dist_right = None
        self.tag_eye = self.args.tag_eye  # which eye to use for april tag ('left' or 'right')
        
        self.init_cameras()
        if self.args.head_camera_type == 0:
            img_shape = (self.head_frame_res[0], self.head_frame_res[1], 3)
            self.shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize, name="pAlad1n3traiT")
            self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)
            self.img_array[:] = np.zeros(img_shape, dtype=np.uint8)
            self.tele_vision = OpenTeleVision(self.head_frame_res, self.shm.name, False)
        elif self.args.head_camera_type == 1:
            img_shape = (self.head_frame_res[0], 2 * self.head_frame_res[1], 3)
            self.shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize, name="SaHlofoLinA")
            self.img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=self.shm.buf)
            self.img_array[:] = np.zeros(img_shape, dtype=np.uint8)
            self.tele_vision = OpenTeleVision(self.head_frame_res, self.shm.name, True)
        else:
            raise NotImplementedError("Not supported camera.")
        
        # command buffer
        self.command = np.zeros(20)  # body: xyzrpy, eef_r: xyzrpy, eef_l: xyzrpy, grippers: 2 angles
        self.command_msg = Float32MultiArray()
        self.command_msg.data = self.command.tolist()
        # eef 6d pose in two types of defined unified frame
        self.eef_uni_command = np.zeros((2, 6))
        
        self.fsm_state_msg = Int32()
        self.fsm_state_msg.data = 0

        # initial values
        self.initial_receive = True
        self.init_body_pos = np.zeros(3)
        self.init_body_rot = np.eye(3)
        self.init_eef_pos = np.zeros((2, 3))
        self.init_eef_rot = np.array([np.eye(3), np.eye(3)])
        self.init_gripper_angles = np.zeros(2)

        self.is_changing_reveive_status = False
        self.rate = rospy.Rate(200)
        
        self.on_reset = False
        self.on_collect = False
        self.on_save_data = False
        
        # self.pinch_gripper_angle_scale = 10.0
        self.pinch_dist_gripper_full_close = 0.02
        self.pinch_dist_gripper_full_open = 0.15
        # self.gripper_full_close_angle = 0.33
        self.gripper_full_close_angle = 0.01
        self.gripper_full_open_angle = 1.8
        self.eef_xyz_scale = 1.0
        
        # to get the ratation matrix of the world frame built by the apple vision pro, relative to the world frame of IssacGym or the real world used by LocoMan
        # first rotate along x axis with 90 degrees, then rotate along z axis with -90 degrees, in the frame of IsaacGym
        self.operator_pov_transform = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ]) @ np.array ([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])
        
        self.last_reset_time = 0.0
        self.last_calib_time = 0.0
        
        self.pause_commands = False

        self.record_episode_counter = 0
        self.current_episode_number = None
        
        # Recording tracking system - read from existing tracker created by test_cameras_avp_optimized.py
        self.base_data_path = "/home/robot/drive/robotool/videos_0903"
        self.task_name = self.args.task_name
        self.task_dir = os.path.join(self.base_data_path, self.task_name)
        self.recording_file = os.path.join(self.task_dir, "recording_tracker.json")
        self.recording_data = self._load_recording_data()
        self.session_id = str(uuid.uuid4())[:8]
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M")
        
        self.reset_trajectories()
        self.init_embodiment_states()
        
        # AprilTag detection setup (for calibrated head pose w.r.t. board)
        self.board_tag_size = self.args.board_tag_size  # meters
        self.apriltag_detector = apriltag.Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            debug=0,
        )
        # Board config: 2x2 tags placed together. Order of tag IDs: TL, TR, BR, BL
        # e.g., "0,1,2,3"
        ids_str = self.args.board_ids
        try:
            ids_list = [int(x.strip()) for x in ids_str.split(',')]
            if len(ids_list) == 4:
                self.board_ids = {
                    'TL': ids_list[0],
                    'TR': ids_list[1],
                    'BR': ids_list[2],
                    'BL': ids_list[3],
                }
            else:
                raise ValueError
        except Exception:
            rospy.logwarn("Invalid --board_ids format. Expected 4 comma-separated integers. Using default 0,1,2,3.")
            self.board_ids = {'TL': 0, 'TR': 1, 'BR': 2, 'BL': 3}
        self.pending_board_capture = False
        self.head_in_board_initial = None  # 4x4 transform
        self.board_in_cam = None  # 4x4 transform from board to camera
        
        # Reference board orientation for consistency across trials
        self.reference_board_orientation = None
        self.reference_file_path = None
        
        # Debug information for calibration
        self.last_normalization_debug = None
        
        # data collection - folder structure will be set dynamically per episode
        if self.args.collect_data:
            self.exp_name = self.args.exp_name
            # exp_data_folder will be set dynamically when starting each episode
            self.exp_data_folder = None
            
        # Publish initialization message
        init_msg = "READY"
        self.collection_status_pub.publish(init_msg)
        rospy.loginfo("Data collection system initialized and ready")

    def _load_recording_data(self):
        """Load recording tracking data from file created by test_cameras_avp_optimized.py"""
        try:
            if os.path.exists(self.recording_file):
                with open(self.recording_file, 'r') as f:
                    data = json.load(f)
                    rospy.loginfo(f"Loaded recording data from {self.recording_file}")
                    return data
            else:
                rospy.loginfo(f"Recording tracker file not found: {self.recording_file}")
                rospy.loginfo("This might be a new task/experiment - will create tracker as needed")
                # Return empty structure - will be populated when needed
                return {"task_name": self.task_name, "experiments": {}, "total_video_time": 0.0, "last_updated": datetime.now().isoformat()}
        except Exception as e:
            rospy.logwarn(f"Error loading recording data: {e}")
            return {"task_name": self.task_name, "experiments": {}, "total_video_time": 0.0, "last_updated": datetime.now().isoformat()}
    
    def _get_next_episode_number(self):
        """Get the next episode number for the current exp name from existing tracker"""
        # Reload the recording data to get the latest episode number
        self.recording_data = self._load_recording_data()
        
        if self.exp_name not in self.recording_data["experiments"]:
            rospy.logwarn(f"Experiment '{self.exp_name}' not found in recording tracker")
            rospy.logwarn("This might be a new experiment - will start with episode 1")
            # Create the experiment entry in the tracker if it doesn't exist
            self._ensure_experiment_in_tracker()
            return 1  # Start with episode 1 for new experiment
        
        next_episode = self.recording_data["experiments"][self.exp_name]["current_episode"] + 1
        rospy.loginfo(f"Next episode number for {self.exp_name}: {next_episode}")
        return next_episode
    
    def _ensure_experiment_in_tracker(self):
        """Ensure the experiment exists in the recording tracker, create if necessary"""
        try:
            # Ensure task directory exists
            os.makedirs(self.task_dir, exist_ok=True)
            
            # If tracker file doesn't exist, create it
            if not os.path.exists(self.recording_file):
                rospy.loginfo(f"Creating new recording tracker file: {self.recording_file}")
                initial_data = {
                    "task_name": self.task_name,
                    "experiments": {},
                    "total_video_time": 0.0,
                    "last_updated": datetime.now().isoformat()
                }
                with open(self.recording_file, 'w') as f:
                    json.dump(initial_data, f, indent=2)
                self.recording_data = initial_data
            
            # Add experiment if it doesn't exist
            if self.exp_name not in self.recording_data["experiments"]:
                rospy.loginfo(f"Adding new experiment '{self.exp_name}' to recording tracker")
                self.recording_data["experiments"][self.exp_name] = {
                    "current_episode": 0,
                    "total_video_time": 0.0,
                    "episode_times": []
                }
                
                # Save updated tracker
                self.recording_data["last_updated"] = datetime.now().isoformat()
                with open(self.recording_file, 'w') as f:
                    json.dump(self.recording_data, f, indent=2)
                
                rospy.loginfo(f"Successfully added experiment '{self.exp_name}' to tracker")
            
        except Exception as e:
            rospy.logerr(f"Error ensuring experiment in tracker: {e}")
            rospy.logerr("Will continue with default episode numbering")

    def auto_calibrate_head_pose(self):
        """Automatically calibrate head pose relative to AprilTag board for each recording episode"""
        rospy.loginfo("Starting automatic head pose calibration for episode...")
        
        # Wait a bit for camera to stabilize (shorter wait for per-episode calibration)
        rospy.sleep(1.0)
        
        # Try to capture head pose multiple times (fewer attempts for faster episode start)
        max_attempts = 5
        for attempt in range(max_attempts):
            rospy.loginfo(f"Episode calibration attempt {attempt + 1}/{max_attempts}")
            
            # Get current frame
            if self.args.head_camera_type == 1:  # Stereo camera
                ret, frame = self.head_cap.read()
                if not ret:
                    rospy.logwarn(f"Failed to read frame on attempt {attempt + 1}")
                    rospy.sleep(0.3)
                    continue
                
                # Resize and split frame
                frame = cv2.resize(frame, (2 * self.head_frame_res[1], self.head_frame_res[0]))
                image_left = frame[:, :self.head_frame_res[1], :]
                image_right = frame[:, self.head_frame_res[1]:, :]
                
                if self.crop_size_w != 0:
                    bgr = np.hstack((image_left[self.crop_size_h:, self.crop_size_w:-self.crop_size_w],
                                    image_right[self.crop_size_h:, self.crop_size_w:-self.crop_size_w]))
                else:
                    bgr = np.hstack((image_left[self.crop_size_h:, :],
                                    image_right[self.crop_size_h:, :]))
                
                self.head_color_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                rospy.logwarn("Auto-calibration only supported for stereo camera (head_camera_type=1)")
                return
            
            # Try to capture head pose
            if self.try_capture_head_in_board():
                rospy.loginfo(f"Episode {self.current_episode_number} head pose calibration successful!")
                rospy.loginfo(f"Head pose relative to board: {self.head_in_board_initial}")
                
                # Log calibration details for this episode
                self.log_episode_calibration_status()
                return
            
            rospy.sleep(0.3)
        
        # If we get here, calibration failed
        rospy.logerr(f"Failed to calibrate head pose for episode {self.current_episode_number}!")
        rospy.logerr("Please ensure the AprilTag board is visible and well-lit")
        rospy.logerr("The system will continue but head pose tracking may be inaccurate")
        rospy.logerr("This episode will not have accurate head pose calibration data")

    def log_episode_calibration_status(self):
        """Log detailed calibration status for the current episode"""
        if self.head_in_board_initial is not None:
            rospy.loginfo(f"=== Episode {self.current_episode_number} Calibration Summary ===")
            rospy.loginfo(f"Head pose in board frame (4x4 transform):")
            rospy.loginfo(f"  Translation: [{self.head_in_board_initial[0,3]:.4f}, {self.head_in_board_initial[1,3]:.4f}, {self.head_in_board_initial[2,3]:.4f}]")
            rospy.loginfo(f"  Rotation matrix:")
            rospy.loginfo(f"    [{self.head_in_board_initial[0,0]:.4f}, {self.head_in_board_initial[0,1]:.4f}, {self.head_in_board_initial[0,2]:.4f}]")
            rospy.loginfo(f"    [{self.head_in_board_initial[1,0]:.4f}, {self.head_in_board_initial[1,1]:.4f}, {self.head_in_board_initial[1,2]:.4f}]")
            rospy.loginfo(f"    [{self.head_in_board_initial[2,0]:.4f}, {self.head_in_board_initial[2,1]:.4f}, {self.head_in_board_initial[2,2]:.4f}]")
            
            if self.board_in_cam is not None:
                rospy.loginfo(f"Board pose in camera frame (4x4 transform):")
                rospy.loginfo(f"  Translation: [{self.board_in_cam[0,3]:.4f}, {self.board_in_cam[1,3]:.4f}, {self.board_in_cam[2,3]:.4f}]")
            
            rospy.loginfo(f"Calibration data will be saved to episode_{self.current_episode_number}.hdf5")
            rospy.loginfo("=" * 50)
        else:
            rospy.logwarn(f"Episode {self.current_episode_number}: No calibration data available")

    def ack_callback(self, msg):
        """Callback for acknowledgment messages from other scripts"""
        ack_msg = msg.data
        
        if ack_msg == "READY":
            self.other_script_ready = True
            rospy.loginfo("Other script acknowledged and is ready")
        elif ack_msg == "STARTED_RECORDING":
            self.waiting_for_ack = False
            self.ack_wait_start_time = None
            self.collection_acknowledged = True
            self.other_script_ready = True
            rospy.loginfo("Other script started recording, proceeding with collection")
            
            # Auto-calibrate head pose at the start of each recording episode
            rospy.loginfo("Starting head pose calibration for new recording episode...")
            self.auto_calibrate_head_pose()
            
            # Trigger capturing board pose at the start of each collection
            self.pending_board_capture = True
            self.head_in_board_initial = None
            
            # Now initialize the embodiment states since we can proceed
            if self.args.collect_data and self.collection_acknowledged:
                self.reset_embodiment_states()
                # The embodiment states will be updated in the next collection cycle
        elif ack_msg == "ERROR":
            rospy.logwarn("Other script reported an error")
            self.waiting_for_ack = False
            self.other_script_ready = False
            self.ack_wait_start_time = None

    def check_ack_timeout(self):
        """Check if we've been waiting too long for acknowledgment"""
        if self.waiting_for_ack and self.ack_wait_start_time is not None:
            elapsed_time = (rospy.Time.now() - self.ack_wait_start_time).to_sec()
            if elapsed_time > self.ack_timeout:
                rospy.logwarn(f"Acknowledgment timeout after {self.ack_timeout} seconds")
                self.waiting_for_ack = False
                self.ack_wait_start_time = None
                self.collection_acknowledged = True  # Allow collection to proceed despite timeout
                return True
        return False

    def reset_pov_transform(self):
        self.operator_pov_transform = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ]) @ np.array ([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])

    def get_gripper_angle_from_pinch_dist(self, pinch_dist):
        scale = (self.gripper_full_open_angle - self.gripper_full_close_angle) / (self.pinch_dist_gripper_full_open - self.pinch_dist_gripper_full_close)
        angle = (pinch_dist - self.pinch_dist_gripper_full_close) * scale + self.gripper_full_close_angle
        return np.clip(angle, self.gripper_full_close_angle, self.gripper_full_open_angle)
    
    def update_manipulate_eef_idx(self):
        if self.args.manipulate_mode == 1:
            # right manipulation
            self.manipulate_eef_idx = [0]
        elif self.args.manipulate_mode == 2:
            # left manipulation
            self.manipulate_eef_idx = [1]
        elif self.args.manipulate_mode == 3:
            # left manipulation
            self.manipulate_eef_idx = [0, 1]

    def sim_view_callback(self, msg):
        self.sim_view_frame = ros_numpy.numpify(msg)
        np.copyto(self.img_array, self.sim_view_frame)
        # self.tele_vision.modify_shared_image(sim_view_image)

    def init_cameras(self):
        import pyrealsense2 as rs
        # initialize all cameras
        self.desired_stream_fps = self.args.desired_stream_fps
        # initialize head camera
        # realsense as head camera
        if self.args.head_camera_type == 0:
            self.head_camera_resolution = (480, 640)
            self.head_frame_res = (480, 640)
            self.head_color_frame = np.zeros((self.head_frame_res[0], self.head_frame_res[1], 3), dtype=np.uint8)
            self.head_cam_pipeline = rs.pipeline()
            self.head_cam_config = rs.config()
            pipeline_wrapper = rs.pipeline_wrapper(self.head_cam_pipeline)
            pipeline_profile = self.head_cam_config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            
            found_rgb = False
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    found_rgb = True
                    break
            if not found_rgb:
                print("a head camera is required")
                exit(0)
            self.head_cam_config.enable_stream(rs.stream.color, self.head_camera_resolution[1], self.head_camera_resolution[0], rs.format.bgr8, 30)
            # start streaming head cam
            self.head_cam_pipeline.start(self.head_cam_config)
            # get intrinsics for AprilTag pose estimation
            try:
                active_profile = self.head_cam_pipeline.get_active_profile()
                color_profile = active_profile.get_stream(rs.stream.color)
                intr = color_profile.as_video_stream_profile().get_intrinsics()
                self.head_intr_param = [intr.fx, intr.fy, intr.ppx, intr.ppy]
            except Exception as e:
                rospy.logwarn(f"Failed to get head camera intrinsics: {e}")
        # stereo rgb camera (dual lens) as the head camera 
        elif self.args.head_camera_type == 1:
            # self.head_camera_resolution: the supported resoltion of the camera, to get the original frame without cropping
            self.head_camera_resolution = (720, 1280) # 720p, the original resolution of the stereo camera is actually 1080p (1080x1920)
            # self.head_view_resolution: the resolution of the images that are seen and recorded
            self.head_view_resolution = (480, 640) # 480p
            self.crop_size_w = 0
            self.crop_size_h = 0
            self.head_frame_res = (self.head_view_resolution[0] - self.crop_size_h, self.head_view_resolution[1] - 2 * self.crop_size_w)
            self.head_color_frame = np.zeros((self.head_frame_res[0], 2 * self.head_frame_res[1], 3), dtype=np.uint8)
            device_map = list_video_devices()
            head_camera_name = "3D USB Camera"
            device_path = find_device_path_by_name(device_map, head_camera_name)
            self.head_cap = cv2.VideoCapture(device_path)
            self.head_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.head_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * self.head_camera_resolution[1])
            self.head_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.head_camera_resolution[0])
            self.head_cap.set(cv2.CAP_PROP_FPS, self.desired_stream_fps)
            # Load stereo intrinsics YAML if provided
            stereo_yaml = self.args.stereo_intrinsics_yaml
            yaml_paths_to_try = [stereo_yaml]
            print("yaml_paths_to_try: ", yaml_paths_to_try)
            # # Fallback: alongside this script
            # try:
            #     script_dir = os.path.dirname(os.path.abspath(__file__))
            #     yaml_paths_to_try.append(os.path.join(script_dir, 'stereo_intrinsics_1.yaml'))
            # except Exception as e:
            #     rospy.logwarn(f"Failed to get head camera intrinsics: {e}")

            loaded = False
            for yaml_path in yaml_paths_to_try:
                if yaml_path and os.path.exists(yaml_path):
                    try:
                        rospy.loginfo(f"Loading stereo intrinsics from: {yaml_path}")
                        with open(yaml_path, 'r') as yf:
                            data = yaml.safe_load(yf)
                        # Expecting keys: left.K, right.K
                        if 'left' in data and 'K' in data['left']:
                            K_l = np.array(data['left']['K'], dtype=np.float32)
                            self.head_intr_param_left = [float(K_l[0,0]), float(K_l[1,1]), float(K_l[0,2]), float(K_l[1,2])]
                            
                            if 'dist' in data['left']:
                                self.head_dist_left = np.array(data['left']['dist'], dtype=np.float32).reshape(-1, 1)
                            rospy.loginfo(f"Left intrinsics fx={self.head_intr_param_left[0]:.3f}, fy={self.head_intr_param_left[1]:.3f}, cx={self.head_intr_param_left[2]:.3f}, cy={self.head_intr_param_left[3]:.3f}")
                        if 'right' in data and 'K' in data['right']:
                            K_r = np.array(data['right']['K'], dtype=np.float32)
                            self.head_intr_param_right = [float(K_r[0,0]), float(K_r[1,1]), float(K_r[0,2]), float(K_r[1,2])]
                            if 'dist' in data['right']:
                                self.head_dist_right = np.array(data['right']['dist'], dtype=np.float32).reshape(-1, 1)
                            rospy.loginfo(f"Right intrinsics fx={self.head_intr_param_right[0]:.3f}, fy={self.head_intr_param_right[1]:.3f}, cx={self.head_intr_param_right[2]:.3f}, cy={self.head_intr_param_right[3]:.3f}")

                        if self.head_intr_param_left is not None and self.head_intr_param_right is not None:
                            rospy.loginfo("Loaded stereo intrinsics for 3D USB Camera.")
                            loaded = True
                            break
                    except Exception as e:
                        rospy.logwarn(f"Failed to load stereo intrinsics YAML at {yaml_path}: {e}")
                else:
                    rospy.logwarn(f"Stereo intrinsics YAML path not found: {yaml_path}")

            if not loaded:
                rospy.logwarn("Stereo intrinsics not loaded. AprilTag head-in-board capture will be skipped until provided.")
        else:
            raise NotImplementedError("Not supported camera.")
        
        # initialize wrist camera/cameras
        if self.args.use_wrist_camera:
            self.wrist_camera_resolution = (720, 1280)
            self.wrist_view_resolution = (480, 640)
            self.wrist_color_frame = np.zeros((self.wrist_view_resolution[0], self.wrist_view_resolution[1], 3), dtype=np.uint8)
            device_map = list_video_devices()
            wrist_camera_name = "Global Shutter Camera"
            device_path = find_device_path_by_name(device_map, wrist_camera_name)
            self.wrist_cap1 = cv2.VideoCapture(device_path)
            self.wrist_cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.wrist_cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.wrist_camera_resolution[1])
            self.wrist_cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.wrist_camera_resolution[0])
            self.wrist_cap1.set(cv2.CAP_PROP_FPS, self.desired_stream_fps)

    def prompt_rendering(self):
        viewer_img = self.head_color_frame
        height = viewer_img.shape[0]
        width = viewer_img.shape[1]
        if self.on_save_data:
            im = PIL.Image.fromarray(self.head_color_frame)
            drawer = PIL.ImageDraw.Draw(im)
            font = PIL.ImageFont.truetype('FreeSans.ttf', size=53)
            drawer.text((width / 8 - 25, (height - 80) / 2), "SAVING DATA", font=font, fill=(10, 255, 10))
            viewer_img = np.array(im)
        elif self.on_reset:
            im = PIL.Image.fromarray(self.head_color_frame)
            drawer = PIL.ImageDraw.Draw(im)
            font = PIL.ImageFont.truetype('FreeSans.ttf', size=53)
            drawer.text((width / 8 - 35, (height - 80) / 2), "PINCH to START", font=font, fill=(255, 63, 63))
            # drawer.text((767, 200), "PINCH to START", font=font, fill=(255, 63, 63))
            viewer_img = np.array(im)
        elif self.on_collect:
            im = PIL.Image.fromarray(self.head_color_frame)
            drawer = PIL.ImageDraw.Draw(im)
            font = PIL.ImageFont.truetype('FreeSans.ttf', size=20)
            
            # Show acknowledgment status
            if self.waiting_for_ack:
                status_text = f"WAITING FOR OTHER SCRIPT... ({self.ack_timeout - (rospy.Time.now() - self.ack_wait_start_time).to_sec():.1f}s)"
                drawer.text((width / 16 - 10, (height - 80) / 8), status_text, font=font, fill=(255, 165, 0))  # Orange color
            
            if len(self.manipulate_eef_idx) == 1:
                drawer.text((width / 16 - 10, (height - 80) / 4), '{:.2f}, {:.2f}, {:.2f} / {:.2f}, {:.2f}, {:.2f}'.format(
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+1],
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+2], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+3],
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+4], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+5]
                        ), font=font, fill=(255, 63, 63))
            elif len(self.manipulate_eef_idx) == 2:
                drawer.text((width / 16 - 10, (height - 80) / 4), '{:.2f}, {:.2f}, {:.2f} / {:.2f}, {:.2f}, {:.2f}'.format(
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+1],
                    self.eef_pose_uni[6*self.manipulate_eef_idx[0]+2], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[1]],
                    self.eef_pose_uni[6*self.manipulate_eef_idx[1]+1], 
                    self.eef_pose_uni[6*self.manipulate_eef_idx[1]+2]
                        ), font=font, fill=(255, 63, 63))
            viewer_img = np.array(im)
        return viewer_img

    def head_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        if self.args.head_camera_type == 0:
            try:
                while not rospy.is_shutdown():
                    start_time = time.time()
                    # handle head camera (realsense) streaming 
                    frames = self.head_cam_pipeline.wait_for_frames()
                    # depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        return
                    head_color_frame = np.asanyarray(color_frame.get_data())
                    head_color_frame = cv2.resize(head_color_frame, (self.head_frame_res[1], self.head_frame_res[0]))
                    self.head_color_frame = cv2.cvtColor(head_color_frame, cv2.COLOR_BGR2RGB)
                    np.copyto(self.img_array, self.head_color_frame)
                    elapsed_time = time.time() - start_time
                    sleep_time = frame_duration - elapsed_time
            finally:
                self.head_cam_pipeline.stop()
        elif self.args.head_camera_type == 1:
            try:
                while not rospy.is_shutdown():
                    start_time = time.time()
                    ret, frame = self.head_cap.read()
                    frame = cv2.resize(frame, (2 * self.head_frame_res[1], self.head_frame_res[0]))
                    image_left = frame[:, :self.head_frame_res[1], :]
                    image_right = frame[:, self.head_frame_res[1]:, :]
                    if self.crop_size_w != 0:
                        bgr = np.hstack((image_left[self.crop_size_h:, self.crop_size_w:-self.crop_size_w],
                                        image_right[self.crop_size_h:, self.crop_size_w:-self.crop_size_w]))
                    else:
                        bgr = np.hstack((image_left[self.crop_size_h:, :],
                                        image_right[self.crop_size_h:, :]))

                    self.head_color_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    viewer_img = self.prompt_rendering()
                    np.copyto(self.img_array, viewer_img)
                    elapsed_time = time.time() - start_time
                    sleep_time = frame_duration - elapsed_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            finally:
                self.head_cap.release()
        else:
            raise NotImplementedError('Not supported camera.')
        
    def wrist_camera_stream_thread(self):
        frame_duration = 1 / self.desired_stream_fps
        try:
            while not rospy.is_shutdown():
                start_time = time.time()
                ret, frame = self.wrist_cap1.read()
                wrist_color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.wrist_color_frame = cv2.resize(wrist_color_frame, (self.wrist_view_resolution[1], self.wrist_view_resolution[0]))
                elapsed_time = time.time() - start_time
                sleep_time = frame_duration - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # print(1/(time.time() - start_time))
        finally:
            self.head_cap.release()

        
    def collect_data(self, event=None):
        # publish teleop commands at a fixed rate 
        # need to check the duration to finish one execution and if a separate thread is needed: takes 0.0002-0.0003s
        self.update_manipulate_eef_idx()
        # left_hand_data: [4, 4]
        left_hand_data = self.tele_vision.left_hand
        # right_hand_data: [4, 4]
        right_hand_data = self.tele_vision.right_hand
        # left_landmarks_data: [25, 3]
        left_landmarks_data = self.tele_vision.left_landmarks
        # right_landmarks_data: [25, 3]
        right_landmarks_data = self.tele_vision.right_landmarks
        # head_data: [4, 4]
        head_data = self.tele_vision.head_matrix
        
        left_wrist_pos = left_hand_data[:3, 3]
        left_wrist_rot = left_hand_data[:3, :3]
        right_wrist_pos = right_hand_data[:3, 3]
        right_wrist_rot = right_hand_data[:3, :3]
        head_pos = head_data[:3, 3]
        head_rot = head_data[:3, :3]        
        
        ### define new head rot and wrist local frame at reset to align with the unified frame which refers to the issacgym world frame
        ### issacgym world frame: x forward, y leftward, z upward
        ### unified frame: z upward (x-y plane horizontal to the ground), x aligns with the forward direction of human head at reset (rotate around z axis using the head yaw)
        head_rot_new = np.zeros_like(head_rot)
        head_rot_new[:, 0] = -head_rot[:, 2]
        head_rot_new[:, 1] = -head_rot[:, 0]
        head_rot_new[:, 2] = head_rot[:, 1]
        # keep the old one to extract head yaw at reset, then rotate the unified frame using the head yaw
        head_rot_old = head_rot
        head_rot = head_rot_new
        
        left_wrist_rot_new = np.zeros_like(left_wrist_rot)
        left_wrist_rot_new[:, 0] = -left_wrist_rot[:, 2]
        left_wrist_rot_new[:, 1] = -left_wrist_rot[:, 0]
        left_wrist_rot_new[:, 2] = left_wrist_rot[:, 1]
        left_wrist_rot = left_wrist_rot_new
        
        right_wrist_rot_new = np.zeros_like(right_wrist_rot)
        right_wrist_rot_new[:, 0] = -right_wrist_rot[:, 2]
        right_wrist_rot_new[:, 1] = -right_wrist_rot[:, 0]
        right_wrist_rot_new[:, 2] = right_wrist_rot[:, 1]
        right_wrist_rot = right_wrist_rot_new

        # thumb to index finger
        left_pinch_dist0 = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[9]) 
        right_pinch_dist0 = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[9]) 
        # thumb to middle finger
        left_pinch_dist1 = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[14]) 
        right_pinch_dist1 = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[14]) 
        # thumb to ring finger
        left_pinch_dist2 = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[19])
        right_pinch_dist2 = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[19])
        # thumb to little finger
        left_pinch_dist3 = np.linalg.norm(left_landmarks_data[4] - left_landmarks_data[24])
        right_pinch_dist3 = np.linalg.norm(right_landmarks_data[4] - right_landmarks_data[24])
        
        # reset
        if left_pinch_dist1 < 0.008 and left_pinch_dist0 > 0.05 and left_pinch_dist2 > 0.05 and left_pinch_dist3 > 0.05 and time.time() - self.last_reset_time > 3.0:
            self.on_reset = True
            self.on_collect = False
            self.last_reset_time = time.time()
            self.initial_receive = True
            self.reset_pov_transform()
            self.flush_trajectory()
            
            # Publish reset message
            reset_msg = "RESET"
            self.collection_status_pub.publish(reset_msg)
            self.collection_acknowledged = False  # Reset acknowledgment state
            rospy.loginfo("Data collection system reset")
            
            print("reset")
        self.command[:] = 0
        # initialize and calibrate
        if self.on_reset:
            # the initialization only begins at a certain signal (e.g., a gesture)
            # the gesture is temporarily designed for single hand manipulation
            if left_pinch_dist0 < 0.008 and left_pinch_dist1 > 0.05 and left_pinch_dist2 > 0.05 and left_pinch_dist3 > 0.05:
                self.on_reset = False
                self.on_collect = True
                
                # Wait for other script to be ready before starting
                if not self.other_script_ready:
                    rospy.logwarn("Other script not ready, waiting for acknowledgment...")
                    self.waiting_for_ack = True
                    self.ack_wait_start_time = rospy.Time.now()
                    return
                
                self.ack_wait_start_time = rospy.Time.now() # debug
                
                # Get next episode number from recording tracker
                episode_num = self._get_next_episode_number()
                self.current_episode_number = episode_num
                self.collection_acknowledged = False  # Reset for new episode
                
                # Create session directory with task/exp structure
                self.exp_data_folder = os.path.join(self.task_dir, f"{self.current_time}_{self.exp_name}_{episode_num}")
                os.makedirs(self.exp_data_folder, exist_ok=True)
                
                # Publish collection start message
                start_msg = f"START:{episode_num}"
                self.collection_status_pub.publish(start_msg)
                rospy.loginfo(f"Started data collection for episode {episode_num} (exp: {self.exp_name})")
                rospy.loginfo(f"Data will be saved to: {self.exp_data_folder}")
                
                # Wait for acknowledgment that other script started recording
                self.waiting_for_ack = True
                rospy.loginfo("Waiting for other script to start recording...")
                
                # Don't proceed with data collection yet - wait for acknowledgment
                return
            else:
                return
        # start collect
        elif self.on_collect:
            # Capture head(camera)-in-board transform once per episode (at start)
            if self.pending_board_capture:
                if self.head_color_frame is not None and self.head_color_frame.size > 0:
                    self.try_capture_head_in_board()
                    # Whether success or not, do not block the rest of collection
                    self.pending_board_capture = False
            # Check for acknowledgment timeout
            if self.check_ack_timeout():
                rospy.logwarn("Proceeding with collection despite timeout")
            
            # Safety check: if we've been waiting too long, proceed anyway
            if self.waiting_for_ack and self.ack_wait_start_time is not None:
                elapsed_time = (rospy.Time.now() - self.ack_wait_start_time).to_sec()
                if elapsed_time > self.ack_timeout * 2:  # Double timeout as safety
                    rospy.logwarn("Safety timeout reached, proceeding with collection")
                    self.waiting_for_ack = False
                    self.ack_wait_start_time = None
                    self.collection_acknowledged = True
            
            if left_pinch_dist3 < 0.008 and left_pinch_dist0 > 0.05 and left_pinch_dist1 > 0.05 and left_pinch_dist2 > 0.05:
                self.pause_commands = True
                
                # Publish pause message
                pause_msg = f"PAUSE:{self.current_episode_number}"
                self.collection_status_pub.publish(pause_msg)
                rospy.loginfo(f"Data collection paused for episode {self.current_episode_number}")
                
                print("Pause sending commands")
                return
            if self.pause_commands and left_pinch_dist0 < 0.008 and left_pinch_dist1 > 0.05 and left_pinch_dist2 > 0.05 and left_pinch_dist3 > 0.05:
                
                # Publish resume message
                resume_msg = f"RESUME:{self.current_episode_number}"
                self.collection_status_pub.publish(resume_msg)
                rospy.loginfo(f"Data collection resumed for episode {self.current_episode_number}")
                
                print("Restart sending commands")
                self.pause_commands = False
            if self.pause_commands:
                return

            if self.args.collect_data and self.collection_acknowledged:
                self.reset_embodiment_states()
                # use the head position (3D) at reset as the origin of the frame
                # body (head) pose [6]
                body_pos_uni = self.operator_pov_transform @ head_pos - self.init_body_pos
                body_rot_uni = self.rot_mat_to_rpy(self.operator_pov_transform @ head_rot)
                body_pose_uni = np.concatenate((body_pos_uni, body_rot_uni))
                self.delta_body_pose_uni = body_pose_uni - self.body_pose_uni
                self.body_pose_uni = body_pose_uni
                # eef pose [6 * 2]
                right_eef_pos_uni = self.operator_pov_transform @ right_wrist_pos - self.init_body_pos
                right_eef_rot_uni = self.rot_mat_to_rpy(self.operator_pov_transform @ right_wrist_rot)
                left_eef_pos_uni = self.operator_pov_transform @ left_wrist_pos - self.init_body_pos
                left_eef_rot_uni = self.rot_mat_to_rpy(self.operator_pov_transform @ left_wrist_rot)
                eef_pose_uni = np.concatenate([right_eef_pos_uni, right_eef_rot_uni, left_eef_pos_uni, left_eef_rot_uni])
                self.delta_eef_pose_uni = eef_pose_uni - self.eef_pose_uni
                self.eef_pose_uni = eef_pose_uni
                # eef relative to body [6 * 2]
                right_eef_to_body_pos = right_eef_pos_uni - body_pos_uni
                right_eef_to_body_rot = self.rot_mat_to_rpy(self.rpy_to_rot_mat(body_rot_uni).T @ self.rpy_to_rot_mat(right_eef_rot_uni))
                left_eef_to_body_pos = left_eef_pos_uni - body_pos_uni
                left_eef_to_body_rot = self.rot_mat_to_rpy(self.rpy_to_rot_mat(body_rot_uni).T @ self.rpy_to_rot_mat(left_eef_rot_uni))
                self.eef_to_body_pose = np.concatenate([right_eef_to_body_pos, right_eef_to_body_rot, left_eef_to_body_pos, left_eef_to_body_rot])
                # simulated gripper (hand as gripper) [1 * 2]
                right_gripper_angle = self.get_gripper_angle_from_pinch_dist(right_pinch_dist0)
                left_gripper_angle = self.get_gripper_angle_from_pinch_dist(left_pinch_dist0)
                gripper_angle = np.array([right_gripper_angle, left_gripper_angle])
                self.delta_gripper_angle = gripper_angle - self.gripper_angle
                self.gripper_angle = gripper_angle
                # hand joints
                self.right_hand_joints = right_landmarks_data.flatten()
                self.left_hand_joints = left_landmarks_data.flatten()
                # main camera pose relative to its initial pose in the unified frame (we assume that the main camera is fixed on the head)
                head_camera_to_init_rot = self.rot_mat_to_rpy(self.init_body_rot.T @ (self.operator_pov_transform @ head_rot))
                self.head_camera_to_init_pose_uni = np.concatenate((body_pos_uni, head_camera_to_init_rot))
                self.update_trajectories()
    
    def init_embodiment_states(self):
        self.body_pose_uni = np.zeros(6)
        self.delta_body_pose_uni = np.zeros(6)
        self.eef_pose_uni = np.zeros(12)
        self.delta_eef_pose_uni = np.zeros(12)
        self.eef_to_body_pose = np.zeros(12)
        self.gripper_angle = np.zeros(2)
        self.delta_gripper_angle = np.zeros(2)
        self.right_hand_joints = np.zeros(75) # (25, 3)
        self.left_hand_joints = np.zeros(75) # (25, 3)
        # camera pose relative to its initial pose in the unified frame
        self.head_camera_to_init_pose_uni = np.zeros(6)

    def reset_embodiment_states(self):
        # need to reset: make sure add the updated state to the trajectory meanwhile keep previous ones unchanged
        # not reset to zeros: copy the value to compute delta
        self.body_pose_uni = self.body_pose_uni.copy()
        self.delta_body_pose_uni = self.delta_body_pose_uni.copy()
        self.eef_pose_uni = self.eef_pose_uni.copy()
        self.delta_eef_pose_uni = self.delta_eef_pose_uni.copy()
        self.eef_to_body_pose = self.eef_to_body_pose.copy()
        self.gripper_angle = self.gripper_angle.copy()
        self.delta_gripper_angle = self.delta_gripper_angle.copy()
        self.right_hand_joints = self.right_hand_joints.copy()
        self.left_hand_joints = self.left_hand_joints.copy()
        # camera pose relative to its initial pose in the unified frame
        self.head_camera_to_init_pose_uni = self.head_camera_to_init_pose_uni.copy()

    def reset_trajectories(self):
        self.main_cam_image_history = []
        self.wrist_cam_image_history = []
        self.body_pose_history = []
        self.delta_body_pose_history = []
        self.eef_pose_history = []
        self.delta_eef_pose_history = []
        self.eef_to_body_pose_history = []
        self.gripper_angle_history = []
        self.delta_gripper_angle_history = []
        self.right_hand_joints_history = []
        self.left_hand_joints_history = []
        self.head_camera_to_init_pose_history = []

    def update_trajectories(self):
        self.main_cam_image_history.append(self.head_color_frame)
        if self.args.use_wrist_camera:
            self.wrist_cam_image_history.append(self.wrist_color_frame) 
        self.body_pose_history.append(self.body_pose_uni)
        self.delta_body_pose_history.append(self.delta_body_pose_uni)
        self.eef_pose_history.append(self.eef_pose_uni)
        self.delta_eef_pose_history.append(self.delta_eef_pose_uni)
        self.eef_to_body_pose_history.append(self.eef_to_body_pose)
        self.gripper_angle_history.append(self.gripper_angle)
        self.delta_gripper_angle_history.append(self.delta_gripper_angle)
        self.right_hand_joints_history.append(self.right_hand_joints)
        self.left_hand_joints_history.append(self.left_hand_joints) 
        self.head_camera_to_init_pose_history.append(self.head_camera_to_init_pose_uni)

    def get_embodiment_masks(self):
        self.img_main_mask = np.array([True])
        self.img_wrist_mask = np.array([False, False])
        if self.args.use_wrist_camera:
            for eef_idx in self.manipulate_eef_idx:
                self.img_wrist_mask[self.manipulate_eef_idx] = True
        self.proprio_body_mask = np.array([True, True, True, True, True, True])
        self.proprio_eef_mask = np.array([False] * 12)
        self.proprio_gripper_mask = np.array([False, False])
        self.proprio_other_mask = np.array([False, False])            
        self.act_body_mask = np.array([True, True, True, True, True, True])
        self.act_eef_mask = np.array([False] * 12)
        self.act_gripper_mask = np.array([False, False])
        for eef_idx in self.manipulate_eef_idx:
            self.proprio_eef_mask[6*eef_idx:6+6*eef_idx] = True
            self.proprio_gripper_mask[eef_idx] = True
            # unsure if we should use the hand joints as priprio
            self.proprio_other_mask[eef_idx] = True
            self.act_eef_mask[6*eef_idx:6+6*eef_idx] = True
            self.act_gripper_mask[eef_idx] = True

    def flush_trajectory(self):
        if self.args.collect_data and len(self.eef_pose_history) > 0:
            self.on_save_data = True
            self.get_embodiment_masks()
            save_path = self.exp_data_folder
            episode_num = self.current_episode_number
            
            # Publish collection end message
            end_msg = f"END:{episode_num}"
            self.collection_status_pub.publish(end_msg)
            rospy.loginfo(f"Finished data collection for episode {episode_num}")

            with h5py.File(os.path.join(save_path, f'episode_{episode_num}.hdf5'), 'a') as f:
                # add the embodiment tag
                f.attrs['embodiment'] = 'human'
                # save observations
                obs_group = f.create_group('observations')
                image_group = obs_group.create_group('images')
                image_group.create_dataset('main', data=self.main_cam_image_history)
                image_group.create_dataset('wrist', data=self.wrist_cam_image_history)
                proprio_group = obs_group.create_group('proprioceptions')
                proprio_group.create_dataset('body', data=self.body_pose_history)
                proprio_group.create_dataset('eef', data=self.eef_pose_history)
                proprio_group.create_dataset('eef_to_body', data=self.eef_to_body_pose_history)
                proprio_group.create_dataset('gripper', data=self.gripper_angle_history)
                proprio_other_group = proprio_group.create_group('other')
                proprio_other_group.create_dataset('right_hand_joints', data=self.right_hand_joints_history)
                proprio_other_group.create_dataset('left_hand_joints', data=self.left_hand_joints_history)
                # save actions
                action_group = f.create_group('actions')
                action_group.create_dataset('body', data=self.body_pose_history)
                action_group.create_dataset('delta_body', data=self.delta_body_pose_history)
                action_group.create_dataset('eef', data=self.eef_pose_history)
                action_group.create_dataset('delta_eef', data=self.delta_eef_pose_history)
                action_group.create_dataset('gripper', data=self.gripper_angle_history)
                action_group.create_dataset('delta_gripper', data=self.delta_gripper_angle_history)
                # save masks
                mask_group = f.create_group('masks')
                mask_group.create_dataset('img_main', data=self.img_main_mask)
                mask_group.create_dataset('img_wrist', data=self.img_wrist_mask)
                mask_group.create_dataset('proprio_body', data=self.proprio_body_mask)
                mask_group.create_dataset('proprio_eef', data=self.proprio_eef_mask)
                mask_group.create_dataset('proprio_gripper', data=self.proprio_gripper_mask)
                mask_group.create_dataset('proprio_other', data=self.proprio_other_mask)
                mask_group.create_dataset('act_body', data=self.act_body_mask)
                mask_group.create_dataset('act_eef', data=self.act_eef_mask)
                mask_group.create_dataset('act_gripper', data=self.act_gripper_mask)
                # save camera poses
                camera_group = f.create_group('camera_poses')
                camera_group.create_dataset('head_camera_to_init', data=self.head_camera_to_init_pose_history)
                
                # Log calibration status for this episode
                rospy.loginfo(f"Saving calibration data for episode {episode_num}:")
                if self.head_in_board_initial is not None:
                    rospy.loginfo(f"  ✓ Head pose calibration available")
                    rospy.loginfo(f"  ✓ Board pose calibration available")
                else:
                    rospy.logwarn(f"  ✗ No head pose calibration data available for episode {episode_num}")
                
                # Save head pose relative to AprilTag board (initial), if available
                if self.head_in_board_initial is not None:
                    camera_group.create_dataset('head_in_board_initial', data=self.head_in_board_initial)
                    camera_group.create_dataset('board_in_cam', data=self.board_in_cam)
                else:
                    # Save an identity as placeholder if not available
                    print("No head_in_board_initial available")
                    camera_group.create_dataset('head_in_board_initial', data=np.eye(4, dtype=np.float32))
                    camera_group.create_dataset('board_in_cam', data=np.eye(4, dtype=np.float32))
            # save videos
            if self.args.save_video:
                h, w, _ = self.main_cam_image_history[0].shape
                freq = self.args.control_freq
                main_cam_video = cv2.VideoWriter(os.path.join(save_path, f'episode_{episode_num}_main_cam_video.mp4'), 
                                                 cv2.VideoWriter_fourcc(*'mp4v'), freq, (w, h))
                
                for image in self.main_cam_image_history:
                    # swap back to bgr for opencv
                    image = image[:, :, [2, 1, 0]] 
                    main_cam_video.write(image)
                main_cam_video.release()
                if self.args.use_wrist_camera:
                    h, w, _ = self.wrist_cam_image_history[0].shape
                    freq = self.args.control_freq
                    wrist_cam_video = cv2.VideoWriter(os.path.join(save_path, f'episode_{episode_num}_wrist_cam_video.mp4'), 
                                                    cv2.VideoWriter_fourcc(*'mp4v'), freq, (w, h))
                    for image in self.wrist_cam_image_history:
                        # swap back to bgr for opencv
                        image = image[:, :, [2, 1, 0]] 
                        wrist_cam_video.write(image)
                    wrist_cam_video.release()
            self.reset_trajectories()
            self.on_save_data = False

    def normalize_board_orientation(self, T_board_in_cam):
        """
        Normalize the board orientation by finding the best 90-degree rotation alignment.
        Tries different 90-degree rotations to find the one that best matches the desired frame:
        - Z-axis points upward (perpendicular to board surface)
        - Y-axis points toward the head direction (in camera view, Y should be downward)
        - X-axis points leftward (in camera view, X should be leftward)
        """
        # Extract rotation matrix and translation
        R = T_board_in_cam[:3, :3]
        t = T_board_in_cam[:3, 3]
        
        # Store original rotation for comparison
        R_original = R.copy()
        
        # Define the desired axes directions in camera frame
        # Z should point up (positive Z), Y should point toward camera (negative Y), X should point left (negative X)
        desired_z = np.array([0, 0, 1])  # Up
        desired_y = np.array([0, -1, 0])  # Toward camera (downward in image)
        desired_x = np.array([-1, 0, 0])  # Leftward
        
        # Generate all possible 90-degree rotations around each axis
        # We'll try rotations around X, Y, Z axes by 0, 90, 180, 270 degrees
        best_R = R.copy()
        best_score = -np.inf
        
        rospy.loginfo("Testing different 90-degree rotations to find best alignment...")
        
        # Test rotations around X-axis (roll)
        for roll_deg in [0, 90, 180, 270]:
            roll_rad = np.radians(roll_deg)
            cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
            Rx = np.array([
                [1, 0, 0],
                [0, cos_r, -sin_r],
                [0, sin_r, cos_r]
            ])
            
            # Test rotations around Y-axis (pitch)
            for pitch_deg in [0, 90, 180, 270]:
                pitch_rad = np.radians(pitch_deg)
                cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
                Ry = np.array([
                    [cos_p, 0, sin_p],
                    [0, 1, 0],
                    [-sin_p, 0, cos_p]
                ])
                
                # Test rotations around Z-axis (yaw)
                for yaw_deg in [0, 90, 180, 270]:
                    yaw_rad = np.radians(yaw_deg)
                    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
                    Rz = np.array([
                        [cos_y, -sin_y, 0],
                        [sin_y, cos_y, 0],
                        [0, 0, 1]
                    ])
                    
                    # Apply the combined rotation
                    R_test = Rz @ Ry @ Rx @ R
                    
                    # Calculate alignment score
                    # Higher score means better alignment with desired directions
                    z_score = np.dot(R_test[:, 2], desired_z)  # Z-axis alignment
                    y_score = np.dot(R_test[:, 1], desired_y)  # Y-axis alignment  
                    x_score = np.dot(R_test[:, 0], desired_x)  # X-axis alignment
                    
                    # Total score (higher is better)
                    total_score = z_score + y_score + x_score
                    
                    # Check if this is the best rotation so far
                    if total_score > best_score:
                        best_score = total_score
                        best_R = R_test.copy()
                        
                        rospy.loginfo(f"New best rotation found: roll={roll_deg}°, pitch={pitch_deg}°, yaw={yaw_deg}°")
                        rospy.loginfo(f"  Score: {total_score:.3f} (Z:{z_score:.3f}, Y:{y_score:.3f}, X:{x_score:.3f})")
        
        # Apply the best rotation
        R = best_R
        
        # Log the final orientation details
        final_z = R[:, 2]
        final_y = R[:, 1]
        final_x = R[:, 0]
        
        rospy.loginfo(f"Final board axes in camera frame after 90-degree rotation optimization:")
        rospy.loginfo(f"  Z-axis (normal): [{final_z[0]:.3f}, {final_z[1]:.3f}, {final_z[2]:.3f}] - should point up")
        rospy.loginfo(f"  Y-axis: [{final_y[0]:.3f}, {final_y[1]:.3f}, {final_y[2]:.3f}] - should point toward camera")
        rospy.loginfo(f"  X-axis: [{final_x[0]:.3f}, {final_x[1]:.3f}, {final_x[2]:.3f}] - should point left")
        
        # Calculate final alignment scores
        final_z_score = np.dot(final_z, desired_z)
        final_y_score = np.dot(final_y, desired_y)
        final_x_score = np.dot(final_x, desired_x)
        final_total_score = final_z_score + final_y_score + final_x_score
        
        rospy.loginfo(f"Final alignment scores: Z:{final_z_score:.3f}, Y:{final_y_score:.3f}, X:{final_x_score:.3f}")
        rospy.loginfo(f"Total alignment score: {final_total_score:.3f}")
        
        # Store debug info for potential saving
        self.last_normalization_debug = {
            'original_R': R_original,
            'final_R': R,
            'best_score': best_score,
            'final_scores': {
                'z': final_z_score,
                'y': final_y_score,
                'x': final_x_score,
                'total': final_total_score
            },
            'desired_axes': {
                'z': desired_z,
                'y': desired_y,
                'x': desired_x
            }
        }
        
        # Verify right-handedness
        cross_product = np.cross(final_x, final_y)
        dot_product = np.dot(cross_product, final_z)
        rospy.loginfo(f"Right-handedness check: dot(cross(X,Y), Z) = {dot_product:.3f} (should be ~1.0)")
        
        # Create the normalized transformation matrix
        T_normalized = np.eye(4, dtype=np.float32)
        T_normalized[:3, :3] = R
        T_normalized[:3, 3] = t
        
        return T_normalized

    def save_reference_board_orientation(self, T_board_in_cam):
        """Save the reference board orientation to ensure consistency across trials"""
        try:
            # Create reference directory if it doesn't exist
            ref_dir = os.path.join(os.path.dirname(self.exp_data_folder), "board_reference")
            os.makedirs(ref_dir, exist_ok=True)
            
            # Save reference orientation
            self.reference_file_path = os.path.join(ref_dir, "reference_board_orientation.npy")
            np.save(self.reference_file_path, T_board_in_cam)
            
            rospy.loginfo(f"Saved reference board orientation to {self.reference_file_path}")
            self.reference_board_orientation = T_board_in_cam.copy()
            
        except Exception as e:
            rospy.logwarn(f"Failed to save reference board orientation: {e}")
    
    def load_reference_board_orientation(self):
        """Load the reference board orientation for consistency"""
        try:
            if self.reference_file_path is None:
                # Try to find existing reference file
                ref_dir = os.path.join(os.path.dirname(self.exp_data_folder), "board_reference")
                ref_file = os.path.join(ref_dir, "reference_board_orientation.npy")
                
                if os.path.exists(ref_file):
                    self.reference_file_path = ref_file
                else:
                    rospy.loginfo("No reference board orientation file found. Will create new reference.")
                    return None
            
            if os.path.exists(self.reference_file_path):
                self.reference_board_orientation = np.load(self.reference_file_path)
                rospy.loginfo(f"Loaded reference board orientation from {self.reference_file_path}")
                return self.reference_board_orientation
            else:
                rospy.loginfo("Reference board orientation file not found.")
                return None
                
        except Exception as e:
            rospy.logwarn(f"Failed to load reference board orientation: {e}")
            return None
    
    def align_to_reference_orientation(self, T_board_in_cam):
        """Align the detected board orientation to the reference orientation"""
        if self.reference_board_orientation is None:
            rospy.loginfo("No reference orientation available. Using normalized orientation.")
            return self.normalize_board_orientation(T_board_in_cam)
        
        # Extract rotation matrices
        R_detected = T_board_in_cam[:3, :3]
        R_reference = self.reference_board_orientation[:3, :3]
        
        # Calculate the difference rotation
        R_diff = R_reference @ R_detected.T
        
        # Extract Euler angles from the difference rotation
        # This tells us how much we need to rotate to align with reference
        angles = self.rotation_matrix_to_euler_angles(R_diff)
        
        rospy.loginfo(f"Board orientation difference from reference: roll={angles[0]:.3f}, pitch={angles[1]:.3f}, yaw={angles[2]:.3f}")
        
        # If the difference is small, use the reference orientation
        angle_threshold = 0.1  # ~5.7 degrees
        if np.all(np.abs(angles) < angle_threshold):
            rospy.loginfo("Detected orientation is close to reference. Using reference orientation.")
            T_aligned = self.reference_board_orientation.copy()
            T_aligned[:3, 3] = T_board_in_cam[:3, 3]  # Keep detected position
            return T_aligned
        else:
            rospy.loginfo("Detected orientation differs significantly from reference. Using normalized orientation.")
            return self.normalize_board_orientation(T_board_in_cam)
    
    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles (roll, pitch, yaw) with ZYX order"""
        sy = -R[2, 0]
        singular_threshold = 1e-6
        cy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

        if cy < singular_threshold:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(sy, cy)
            z = 0
        else:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(sy, cy)
            z = np.arctan2(R[1, 0], R[0, 0])

        return np.array([x, y, z])

    def try_capture_head_in_board(self):
        """Detect AprilTag board in the current head image and compute head(camera) pose in board frame.
        Uses reference board orientation for consistency across trials.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            img_rgb = self.head_color_frame
            if img_rgb is None:
                return False
            # Prepare grayscale and intrinsics depending on camera type
            if self.args.head_camera_type == 0:
                # Single RGB image
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                intr = self.head_intr_param
                if intr is None:
                    rospy.logwarn("Head camera intrinsics not available: cannot compute head_in_board_initial.")
                    return False
            elif self.args.head_camera_type == 1:
                # Stereo: split left/right
                H, W, _ = img_rgb.shape
                half_W = W // 2
                left_rgb = img_rgb[:, :half_W]
                right_rgb = img_rgb[:, half_W:]
                
                if self.tag_eye == 'right':
                    eye_rgb = right_rgb
                    intr = self.head_intr_param_right
                else:
                    eye_rgb = left_rgb
                    intr = self.head_intr_param_left
                if intr is None:
                    rospy.logwarn("Stereo intrinsics not available or not set")
                    intr = [199.90196228027344, 266.3130187988281, 318.70196533203125, 244.07846069335938]
                    # return
                gray = cv2.cvtColor(eye_rgb, cv2.COLOR_RGB2GRAY)
            else:
                return False

            # Load reference board orientation if available
            self.load_reference_board_orientation()

            if self.args.head_camera_type == 0:
                # Use single tag pose estimation
                detections = self.apriltag_detector.detect(gray, True, intr, self.board_tag_size)
                if len(detections) == 0:
                    rospy.logwarn("No AprilTag detected on board at collection start.")
                    return False
                det = max(detections, key=lambda d: getattr(d, 'decision_margin', 0))
                R = np.asarray(det.pose_R).astype(np.float32)
                t = np.asarray(det.pose_t).reshape(3).astype(np.float32)
                T_board_in_cam = np.eye(4, dtype=np.float32)
                T_board_in_cam[:3, :3] = R
                T_board_in_cam[:3, 3] = t
                
                # Align to reference orientation for consistency
                T_board_in_cam = self.align_to_reference_orientation(T_board_in_cam)
                
                # Check board distance before accepting calibration
                board_distance = np.linalg.norm(T_board_in_cam[:3, 3])
                max_board_distance = self.args.max_board_distance  # Configurable maximum distance
                
                if board_distance > max_board_distance:
                    rospy.logwarn(f"Board too far for calibration: {board_distance:.3f}m > {max_board_distance}m")
                    rospy.logwarn("Please move the board closer to the camera for accurate calibration")
                    return False
                
                rospy.loginfo(f"Board distance OK for calibration: {board_distance:.3f}m <= {max_board_distance}m")
                
                T_cam_in_board = np.linalg.inv(T_board_in_cam)
                self.head_in_board_initial = T_cam_in_board
                
                # Save reference orientation if this is the first successful detection
                if self.reference_board_orientation is None:
                    self.save_reference_board_orientation(T_board_in_cam)
                
                # Save debug calibration frame
                self.save_calibration_debug_frame(img_rgb, detections, T_board_in_cam, intr, "single_tag")
                
                rospy.loginfo("Captured head_in_board_initial transform (single-tag).")
                return True
            else:
                # Stereo: aggregate 4 tags for robust board pose via PnP
                # Try multiple frames to get all 4 tags, or use fallback methods
                max_frame_attempts = self.args.max_calibration_frames  # Try up to max_calibration_frames
                best_detections = None
                best_tag_count = 0
                best_eye_rgb = eye_rgb  # Store the best frame for debug
                
                for frame_attempt in range(max_frame_attempts):
                    # Get fresh frame
                    if frame_attempt > 0:
                        rospy.loginfo(f"Frame attempt {frame_attempt + 1}/{max_frame_attempts} - trying to get more tags")
                        rospy.sleep(0.1)  # Small delay between attempts
                        
                        # Get new frame for this attempt
                        if self.args.head_camera_type == 1:  # Stereo camera
                            ret, frame = self.head_cap.read()
                            if not ret:
                                continue
                            
                            # Resize and split frame
                            frame = cv2.resize(frame, (2 * self.head_frame_res[1], self.head_frame_res[0]))
                            image_left = frame[:, :self.head_frame_res[1], :]
                            image_right = frame[:, self.head_frame_res[1]:, :]
                            
                            if self.tag_eye == 'left':
                                eye_rgb = image_left
                            else:
                                eye_rgb = image_right
                            
                            gray = cv2.cvtColor(eye_rgb, cv2.COLOR_RGB2GRAY)
                        else:
                            continue
                    
                    # Detect tags in current frame
                    detections = self.apriltag_detector.detect(gray, camera_params=None, tag_size=self.board_tag_size)
                    
                    # Count how many board tags we found
                    board_tag_count = 0
                    for d in detections:
                        tag_id = getattr(d, 'tag_id', None)
                        if tag_id in self.board_ids.values():
                            board_tag_count += 1
                    
                    rospy.loginfo(f"Frame {frame_attempt + 1}: Found {board_tag_count}/4 board tags")
                    
                    # Update best detection if we found more tags
                    if board_tag_count > best_tag_count:
                        best_tag_count = board_tag_count
                        best_detections = detections.copy()
                        best_eye_rgb = eye_rgb.copy()
                        
                        # If we have all 4 tags, we can stop
                        if board_tag_count == 4:
                            rospy.loginfo("All 4 board tags detected! Proceeding with calibration.")
                            break
                
                # Use the best detection we found
                detections = best_detections
                eye_rgb = best_eye_rgb
                if detections is None or len(detections) == 0:
                    rospy.logwarn("No AprilTags detected after multiple frame attempts.")
                    return False
                
                rospy.loginfo(f"Using detection with {best_tag_count}/4 board tags")
                
                # Build mapping from tag_id to nominal center in board frame
                s = float(self.board_tag_size)
                half = s * 0.5
                # Centers for TL, TR, BR, BL (touching board, no gap)
                centers = {
                    self.board_ids['TL']: (-half, -half),
                    self.board_ids['TR']: ( half, -half),
                    self.board_ids['BR']: ( half,  half),
                    self.board_ids['BL']: (-half,  half),
                }

                obj_pts = []
                img_pts = []
                detected_tag_ids = []
                
                for d in detections:
                    tag_id = getattr(d, 'tag_id', None)
                    if tag_id not in centers:
                        continue
                    detected_tag_ids.append(tag_id)
                    cx, cy = centers[tag_id]
                    # Corners order: TL, TR, BR, BL
                    obj_corners = np.array([
                        [cx - half, cy - half, 0.0],  # TL
                        [cx + half, cy - half, 0.0],  # TR
                        [cx + half, cy + half, 0.0],  # BR
                        [cx - half, cy + half, 0.0],  # BL
                    ], dtype=np.float32)
                    img_corners = np.asarray(d.corners, dtype=np.float32).reshape(4, 2)
                    obj_pts.append(obj_corners)
                    img_pts.append(img_corners)

                rospy.loginfo(f"Using tags: {detected_tag_ids} for calibration")
                
                # Handle different numbers of detected tags
                if len(obj_pts) < 2:
                    rospy.logwarn("Insufficient tags for board PnP. Need at least 2.")
                    return False
                
                # If we have fewer than 4 tags, try to estimate missing tag positions
                if len(detected_tag_ids) < 4:
                    rospy.logwarn(f"Only {len(detected_tag_ids)}/4 tags detected. Attempting to estimate missing tags.")
                    
                    # Try to estimate missing tag positions based on detected ones
                    estimated_obj_pts, estimated_img_pts = self.estimate_missing_tags(
                        detected_tag_ids, obj_pts, img_pts, centers, self.board_tag_size
                    )
                    
                    if estimated_obj_pts is not None:
                        rospy.loginfo("Successfully estimated missing tag positions")
                        obj_pts = estimated_obj_pts
                        img_pts = estimated_img_pts
                    else:
                        rospy.logwarn("Could not estimate missing tags, proceeding with available tags")

                obj_pts = np.concatenate(obj_pts, axis=0)
                img_pts = np.concatenate(img_pts, axis=0).reshape(-1, 1, 2)

                # Build camera matrix and distortion for chosen eye
                fx, fy, cx, cy = intr
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                dist = self.head_dist_right if self.tag_eye == 'right' else self.head_dist_left
                if dist is None:
                    dist = np.zeros((5, 1), dtype=np.float32)

                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    rospy.logwarn("solvePnP failed for board pose.")
                    return False
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.reshape(3)
                T_board_in_cam = np.eye(4, dtype=np.float32)
                T_board_in_cam[:3, :3] = R.astype(np.float32)
                T_board_in_cam[:3, 3] = t.astype(np.float32)
                
                # Align to reference orientation for consistency
                T_board_in_cam = self.align_to_reference_orientation(T_board_in_cam)
                
                # Check board distance before accepting calibration
                board_distance = np.linalg.norm(T_board_in_cam[:3, 3])
                max_board_distance = self.args.max_board_distance  # Configurable maximum distance
                
                if board_distance > max_board_distance:
                    rospy.logwarn(f"Board too far for calibration: {board_distance:.3f}m > {max_board_distance}m")
                    rospy.logwarn("Please move the board closer to the camera for accurate calibration")
                    return False
                
                rospy.loginfo(f"Board distance OK for calibration: {board_distance:.3f}m <= {max_board_distance}m")
                
                print(T_board_in_cam)
                T_cam_in_board = np.linalg.inv(T_board_in_cam)
                print(T_cam_in_board)
                self.head_in_board_initial = T_cam_in_board
                self.board_in_cam = T_board_in_cam
                
                # Save reference orientation if this is the first successful detection
                if self.reference_board_orientation is None:
                    self.save_reference_board_orientation(T_board_in_cam)
                
                # Save debug calibration frame
                self.save_calibration_debug_frame(eye_rgb, detections, T_board_in_cam, intr, "multi_tag_pnp")
                
                rospy.loginfo("Captured head_in_board_initial transform (multi-tag PnP).")
                return True
        except Exception as e:
            rospy.logwarn(f"Failed to capture head_in_board_initial: {e}")
            return False
    
    def save_calibration_debug_frame(self, img_rgb, detections, T_board_in_cam, intr, method_name):
        """Save a debug frame showing the calibration results with drawn axes and information.
        
        Args:
            img_rgb: RGB image used for calibration
            detections: List of AprilTag detections
            T_board_in_cam: 4x4 transformation matrix from board to camera
            intr: Camera intrinsics [fx, fy, cx, cy]
            method_name: Name of the calibration method used
        """
        try:
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw AprilTag detections
            for detection in detections:
                # Draw tag corners
                corners = detection.corners.astype(int)
                cv2.polylines(img_bgr, [corners], True, (0, 255, 0), 2)
                
                # Draw tag ID
                center = np.mean(corners, axis=0).astype(int)
                cv2.putText(img_bgr, f"ID:{detection.tag_id}", 
                           (center[0], center[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw decision margin if available
                if hasattr(detection, 'decision_margin'):
                    cv2.putText(img_bgr, f"DM:{detection.decision_margin:.2f}", 
                               (center[0], center[1] + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Add detection status (detected vs estimated)
                if hasattr(detection, 'is_estimated') and detection.is_estimated:
                    cv2.putText(img_bgr, "ESTIMATED", 
                               (center[0], center[1] + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)  # Orange color
                else:
                    cv2.putText(img_bgr, "DETECTED", 
                               (center[0], center[1] + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # Green color
            
            # Draw coordinate axes at board center
            if T_board_in_cam is not None:
                # Extract rotation and translation
                R = T_board_in_cam[:3, :3]
                t = T_board_in_cam[:3, 3]
                
                # Define axis endpoints in board frame (in meters)
                # Use a smaller axis length that's proportional to board size
                axis_length = min(self.board_tag_size * self.args.axis_length_factor, 0.05)  # configurable factor or 5cm max
                axis_points = np.array([
                    [0, 0, 0],           # Origin
                    [axis_length, 0, 0], # X-axis
                    [0, axis_length, 0], # Y-axis  
                    [0, 0, axis_length]  # Z-axis
                ], dtype=np.float32)
                
                # Transform to camera frame
                axis_points_cam = R @ axis_points.T + t.reshape(3, 1)
                
                # Project to image coordinates with proper distortion handling
                fx, fy, cx, cy = intr
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                
                # Get distortion coefficients for the appropriate eye
                if self.args.head_camera_type == 1:  # Stereo camera
                    dist = self.head_dist_right if self.tag_eye == 'right' else self.head_dist_left
                else:
                    dist = None
                
                if dist is None:
                    dist = np.zeros((5, 1), dtype=np.float32)
                
                # Project 3D points to 2D with distortion correction
                img_points = []
                axis_points_cam_array = axis_points_cam.T.reshape(-1, 1, 3)
                
                # Use cv2.projectPoints for proper distortion handling
                projected_points, _ = cv2.projectPoints(axis_points_cam_array, np.zeros(3), np.zeros(3), K, dist)
                projected_points = projected_points.reshape(-1, 2)
                
                # Validate projected points
                img_h, img_w = img_bgr.shape[:2]
                valid_points = []
                
                for i, point_2d in enumerate(projected_points):
                    x, y = point_2d
                    
                    # Check if point is in front of camera (Z > 0)
                    if axis_points_cam[2, i] > 0:
                        # Check if point is within image bounds
                        if 0 <= x < img_w and 0 <= y < img_h:
                            valid_points.append(point_2d.astype(int))
                        else:
                            rospy.logwarn(f"Axis point {i} projected outside image bounds: ({x:.1f}, {y:.1f})")
                            valid_points.append(None)
                    else:
                        rospy.logwarn(f"Axis point {i} is behind camera: Z = {axis_points_cam[2, i]:.3f}")
                        valid_points.append(None)
                
                img_points = valid_points
                
                # Check board distance from camera
                board_distance = np.linalg.norm(t)
                max_board_distance = self.args.max_board_distance  # Configurable maximum distance
                
                if board_distance > max_board_distance:
                    rospy.logwarn(f"Board too far from camera: {board_distance:.3f}m > {max_board_distance}m")
                    cv2.putText(img_bgr, f"BOARD TOO FAR: {board_distance:.2f}m", 
                               (img_w//2 - 150, img_h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    rospy.loginfo(f"Board distance OK: {board_distance:.3f}m <= {max_board_distance}m")
                
                # Log axis projection debug info
                rospy.loginfo(f"Axis projection debug:")
                rospy.loginfo(f"  Board center in camera: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
                rospy.loginfo(f"  Board distance: {board_distance:.3f}m")
                rospy.loginfo(f"  Axis length: {axis_length:.3f}m")
                rospy.loginfo(f"  Projected points: {[p.tolist() if p is not None else None for p in img_points]}")
                
                # Draw axes if we have valid projections
                if len(img_points) == 4 and all(p is not None for p in img_points):
                    origin = img_points[0]
                    x_end = img_points[1]
                    y_end = img_points[2]
                    z_end = img_points[3]
                    
                    # Draw X-axis (red)
                    cv2.arrowedLine(img_bgr, tuple(origin), tuple(x_end), (0, 0, 255), 3)
                    cv2.putText(img_bgr, "X", tuple(x_end), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Draw Y-axis (green)
                    cv2.arrowedLine(img_bgr, tuple(origin), tuple(y_end), (0, 255, 0), 3)
                    cv2.putText(img_bgr, "Y", tuple(y_end), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw Z-axis (blue)
                    cv2.arrowedLine(img_bgr, tuple(origin), tuple(z_end), (255, 0, 0), 3)
                    cv2.putText(img_bgr, "Z", tuple(z_end), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Draw origin point
                    cv2.circle(img_bgr, tuple(origin), 5, (255, 255, 255), -1)
                    
                    # Add axis info text
                    cv2.putText(img_bgr, f"Axis Length: {axis_length*1000:.0f}mm", 
                               (10, img_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                else:
                    rospy.logwarn("Could not draw axes - invalid projections detected")
                    # Draw a warning on the image
                    cv2.putText(img_bgr, "AXES PROJECTION FAILED", 
                               (img_w//2 - 100, img_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add text information overlay
            # Top-left corner info
            info_lines = [
                f"Method: {method_name}",
                f"Episode: {self.current_episode_number if self.current_episode_number else 'N/A'}",
                f"Camera: {'Stereo' if self.args.head_camera_type == 1 else 'RealSense'}",
                f"Eye: {self.tag_eye}",
                f"Tag Size: {self.board_tag_size}m",
                f"Board IDs: {self.args.board_ids}"
            ]
            
            # Add distance information if available
            if T_board_in_cam is not None:
                board_distance = np.linalg.norm(T_board_in_cam[:3, 3])
                distance_status = "OK" if board_distance <= self.args.max_board_distance else "TOO FAR"
                distance_color = (0, 255, 0) if board_distance <= self.args.max_board_distance else (0, 0, 255)
                info_lines.extend([
                    f"Board Distance: {board_distance:.3f}m ({distance_status})",
                    f"Max Distance: {self.args.max_board_distance}m"
                ])
            
            y_offset = 30
            for line in info_lines:
                cv2.putText(img_bgr, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            # Add transformation matrix info (bottom-right)
            if T_board_in_cam is not None:
                matrix_lines = [
                    "Board in Camera Transform:",
                    f"Translation: [{T_board_in_cam[0,3]:.3f}, {T_board_in_cam[1,3]:.3f}, {T_board_in_cam[2,3]:.3f}]",
                    f"Rotation Matrix:",
                    f"  [{T_board_in_cam[0,0]:.3f}, {T_board_in_cam[0,1]:.3f}, {T_board_in_cam[0,2]:.3f}]",
                    f"  [{T_board_in_cam[1,0]:.3f}, {T_board_in_cam[1,1]:.3f}, {T_board_in_cam[1,2]:.3f}]",
                    f"  [{T_board_in_cam[2,0]:.3f}, {T_board_in_cam[2,1]:.3f}, {T_board_in_cam[2,2]:.3f}]"
                ]
                
                # Add normalization scores if available
                if hasattr(self, 'last_normalization_debug') and self.last_normalization_debug is not None:
                    scores = self.last_normalization_debug['final_scores']
                    matrix_lines.extend([
                        "",
                        "Normalization Scores:",
                        f"Z-axis alignment: {scores['z']:.3f}",
                        f"Y-axis alignment: {scores['y']:.3f}",
                        f"X-axis alignment: {scores['x']:.3f}",
                        f"Total score: {scores['total']:.3f}"
                    ])
                
                # Calculate position for bottom-right text
                img_h, img_w = img_bgr.shape[:2]
                y_offset = img_h - 20
                for line in reversed(matrix_lines):
                    text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    x_pos = img_w - text_size[0] - 10
                    cv2.putText(img_bgr, line, (x_pos, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset -= 15
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(img_bgr, timestamp, (10, img_bgr.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Save the debug frame
            if self.args.collect_data and self.args.save_calibration_debug and self.current_episode_number:
                # Create debug directory
                debug_dir = os.path.join(self.exp_data_folder, "calibration_debug")
                os.makedirs(debug_dir, exist_ok=True)
                
                # Save frame
                filename = f"episode_{self.current_episode_number}_{method_name}_calibration_debug.jpg"
                filepath = os.path.join(debug_dir, filename)
                cv2.imwrite(filepath, img_bgr)
                
                rospy.loginfo(f"Saved calibration debug frame: {filepath}")
                
                # Also save the raw transformation data as text
                txt_filename = f"episode_{self.current_episode_number}_{method_name}_calibration_data.txt"
                txt_filepath = os.path.join(debug_dir, txt_filename)
                
                with open(txt_filepath, 'w') as f:
                    f.write(f"Calibration Debug Data for Episode {self.current_episode_number}\n")
                    f.write(f"Method: {method_name}\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Camera Type: {'Stereo' if self.args.head_camera_type == 1 else 'RealSense'}\n")
                    f.write(f"Eye Used: {self.tag_eye}\n")
                    f.write(f"Tag Size: {self.board_tag_size}m\n")
                    f.write(f"Board IDs: {self.args.board_ids}\n")
                    f.write(f"Intrinsics: {intr}\n")
                    
                    # Add distance information
                    if T_board_in_cam is not None:
                        board_distance = np.linalg.norm(T_board_in_cam[:3, 3])
                        f.write(f"Board Distance: {board_distance:.3f}m\n")
                        f.write(f"Max Allowed Distance: {self.args.max_board_distance}m\n")
                        f.write(f"Distance Status: {'OK' if board_distance <= self.args.max_board_distance else 'TOO FAR'}\n")
                    f.write(f"\nBoard in Camera Transform (4x4):\n")
                    f.write(str(T_board_in_cam))
                    f.write(f"\n\nCamera in Board Transform (4x4):\n")
                    f.write(str(np.linalg.inv(T_board_in_cam)))
                    
                    if self.reference_board_orientation is not None:
                        f.write(f"\n\nReference Board Orientation (4x4):\n")
                        f.write(str(self.reference_board_orientation))
                    
                    f.write(f"\n\nDetections:\n")
                    for i, det in enumerate(detections):
                        f.write(f"Detection {i}:\n")
                        f.write(f"  Tag ID: {det.tag_id}\n")
                        f.write(f"  Decision Margin: {getattr(det, 'decision_margin', 'N/A')}\n")
                        f.write(f"  Corners: {det.corners}\n")
                        if hasattr(det, 'pose_R') and hasattr(det, 'pose_t'):
                            f.write(f"  Pose R: {det.pose_R}\n")
                            f.write(f"  Pose t: {det.pose_t}\n")
                    
                    # Add normalization debug info if available
                    if hasattr(self, 'last_normalization_debug') and self.last_normalization_debug is not None:
                        f.write(f"\n\nNormalization Debug Info:\n")
                        f.write(f"Best Score: {self.last_normalization_debug['best_score']:.3f}\n")
                        f.write(f"Final Scores: Z={self.last_normalization_debug['final_scores']['z']:.3f}, ")
                        f.write(f"Y={self.last_normalization_debug['final_scores']['y']:.3f}, ")
                        f.write(f"X={self.last_normalization_debug['final_scores']['x']:.3f}\n")
                        f.write(f"Total Score: {self.last_normalization_debug['final_scores']['total']:.3f}\n")
                        f.write(f"Desired Axes: Z={self.last_normalization_debug['desired_axes']['z']}, ")
                        f.write(f"Y={self.last_normalization_debug['desired_axes']['y']}, ")
                        f.write(f"X={self.last_normalization_debug['desired_axes']['x']}\n")
                        f.write(f"\nOriginal Rotation Matrix:\n")
                        f.write(str(self.last_normalization_debug['original_R']))
                        f.write(f"\n\nFinal Rotation Matrix:\n")
                        f.write(str(self.last_normalization_debug['final_R']))
                
                rospy.loginfo(f"Saved calibration debug data: {txt_filepath}")
                
        except Exception as e:
            rospy.logwarn(f"Failed to save calibration debug frame: {e}")
            import traceback
            traceback.print_exc()
    
    def estimate_missing_tags(self, detected_tag_ids, obj_pts, img_pts, centers, tag_size):
        """Estimate positions of missing AprilTags based on detected ones.
        
        Args:
            detected_tag_ids: List of detected tag IDs
            obj_pts: List of 3D object points for detected tags
            img_pts: List of 2D image points for detected tags
            centers: Dictionary mapping tag IDs to their nominal centers
            tag_size: Size of each AprilTag
            
        Returns:
            tuple: (estimated_obj_pts, estimated_img_pts) or (None, None) if estimation fails
        """
        try:
            # If we have 3 tags, we can estimate the 4th one
            if len(detected_tag_ids) == 3:
                rospy.loginfo("Attempting to estimate 4th tag from 3 detected tags")
                
                # Find which tag is missing
                all_tag_ids = list(centers.keys())
                missing_tag_id = None
                for tag_id in all_tag_ids:
                    if tag_id not in detected_tag_ids:
                        missing_tag_id = tag_id
                        break
                
                if missing_tag_id is None:
                    rospy.logwarn("Could not identify missing tag ID")
                    return None, None
                
                # Get the nominal center of the missing tag
                missing_center = centers[missing_tag_id]
                
                # Estimate the missing tag's image position using geometric relationships
                # This is a simple approach - in practice, you might want more sophisticated estimation
                
                # For now, we'll use a basic geometric estimation
                # Calculate the center of the detected tags in image space
                detected_img_centers = []
                for img_pt in img_pts:
                    center_pt = np.mean(img_pt.reshape(-1, 2), axis=0)
                    detected_img_centers.append(center_pt)
                
                # Calculate the center of detected tags in object space
                detected_obj_centers = []
                for obj_pt in obj_pts:
                    center_pt = np.mean(obj_pt.reshape(-1, 3), axis=0)
                    detected_obj_centers.append(center_pt)
                
                # Estimate the transformation from object to image space using detected tags
                if len(detected_img_centers) >= 3:
                    # Use PnP to estimate pose from detected tags
                    detected_img_centers_array = np.array(detected_img_centers, dtype=np.float32).reshape(-1, 1, 2)
                    detected_obj_centers_array = np.array(detected_obj_centers, dtype=np.float32).reshape(-1, 1, 3)
                    
                    # Simple camera matrix (you might want to use actual intrinsics)
                    K = np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], dtype=np.float32)
                    
                    ok, rvec, tvec = cv2.solvePnP(detected_obj_centers_array, detected_img_centers_array, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
                    
                    if ok:
                        # Transform the missing tag's 3D position to image space
                        missing_obj_center = np.array([missing_center[0], missing_center[1], 0.0], dtype=np.float32)
                        
                        # Apply the estimated transformation
                        R, _ = cv2.Rodrigues(rvec)
                        missing_img_center = K @ (R @ missing_obj_center.reshape(3, 1) + tvec)
                        missing_img_center = missing_img_center[:2] / missing_img_center[2]
                        
                        # Create the estimated tag corners (assuming square tag)
                        half = tag_size * 0.5
                        estimated_obj_corners = np.array([
                            [missing_center[0] - half, missing_center[1] - half, 0.0],  # TL
                            [missing_center[0] + half, missing_center[1] - half, 0.0],  # TR
                            [missing_center[0] + half, missing_center[1] + half, 0.0],  # BR
                            [missing_center[0] - half, missing_center[1] + half, 0.0],  # BL
                        ], dtype=np.float32)
                        
                        # Estimate image corners by offsetting from center
                        estimated_img_corners = []
                        for corner_obj in estimated_obj_corners:
                            corner_img = K @ (R @ corner_obj.reshape(3, 1) + tvec)
                            corner_img = corner_img[:2] / corner_img[2]
                            estimated_img_corners.append(corner_img.reshape(1, 2))
                        
                        estimated_img_corners = np.array(estimated_img_corners, dtype=np.float32)
                        
                        # Add to our lists
                        obj_pts.append(estimated_obj_corners)
                        img_pts.append(estimated_img_corners)
                        
                        rospy.loginfo(f"Successfully estimated missing tag {missing_tag_id}")
                        return obj_pts, img_pts
                
                rospy.logwarn("Failed to estimate missing tag position")
                return None, None
            
            # If we have 2 tags, we can try to estimate the other 2
            elif len(detected_tag_ids) == 2:
                rospy.loginfo("Only 2 tags detected - estimation of 2 missing tags is challenging")
                rospy.logwarn("Proceeding with 2 tags only - calibration may be less accurate")
                return None, None
            
            else:
                rospy.logwarn(f"Unexpected number of detected tags: {len(detected_tag_ids)}")
                return None, None
                
        except Exception as e:
            rospy.logwarn(f"Error estimating missing tags: {e}")
            return None, None
    
    
    def run(self):
        head_camera_streaming_thread = threading.Thread(target=self.head_camera_stream_thread, daemon=True)
        head_camera_streaming_thread.start()
        if self.args.use_wrist_camera:
            wrist_camera_streaming_thread = threading.Thread(target=self.wrist_camera_stream_thread, daemon=True)
            wrist_camera_streaming_thread.start()
        rospy.Timer(rospy.Duration(1.0 / self.args.control_freq), self.collect_data)
        rospy.spin()

    def rot_mat_to_rpy(self, R):
        """
        Convert a rotation matrix to Euler angles (roll, pitch, yaw) with ZYX order.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            numpy array of [roll, pitch, yaw] in radians
        """
        sy = -R[2, 0]
        singular_threshold = 1e-6
        cy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

        if cy < singular_threshold:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(sy, cy)
            z = 0
        else:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(sy, cy)
            z = np.arctan2(R[1, 0], R[0, 0])

        return np.array([x, y, z])

    def rot_mat_to_rpy_zxy(self, R):
        """
        Convert a rotation matrix (RzRxRy) to Euler angles with ZXY order (first rotate with y, then x, then z).
        This method is more numerically stable, especially near singularities.
        """
        sx = R[2, 1]
        singular_threshold = 1e-6
        cx = np.sqrt(R[2, 0]**2 + R[2, 2]**2)

        if cx < singular_threshold:
            x = np.arctan2(sx, cx)
            y = np.arctan2(R[0, 2], R[0, 0])
            z = 0
        else:
            x = np.arctan2(sx, cx)
            y = np.arctan2(-R[2, 0], R[2, 2])
            z = np.arctan2(-R[0, 1], R[1, 1])

        return np.array([x, y, z])

    def rpy_to_rot_mat(self, rpy):
        """
        Convert Euler angles (roll, pitch, yaw) with ZYX order to a rotation matrix.
        
        Args:
            rpy: numpy array of [roll, pitch, yaw] in radians
            
        Returns:
            3x3 rotation matrix
        """
        x, y, z = rpy
        cx, sx = np.cos(x), np.sin(x)
        cy, sy = np.cos(y), np.sin(y)
        cz, sz = np.cos(z), np.sin(z)

        R = np.array([
            [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
            [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
            [-sy, cy * sx, cx * cy]
        ])
        return R

    def close(self):
        # Publish shutdown message
        shutdown_msg = "SHUTDOWN"
        self.collection_status_pub.publish(shutdown_msg)
        rospy.loginfo("Data collection system shutting down")
        
        self.shm.close()
        self.shm.unlink()       
        print('clean up shared memory')

def signal_handler(sig, frame):
    print('pressed Ctrl+C! exiting...')
    rospy.signal_shutdown('Ctrl+C pressed')
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(prog="teleoperation with apple vision pro and vuer")
    parser.add_argument("--head_camera_type", type=int, default=1, help="0=realsense, 1=stereo rgb camera")
    parser.add_argument("--use_wrist_camera", type=str2bool, default=False, help="whether to use wrist camera")
    parser.add_argument("--desired_stream_fps", type=int, default=25, help="desired camera streaming fps to vuer")
    parser.add_argument("--control_freq", type=int, default=25, help="frequency to record human data")
    parser.add_argument("--collect_data", type=str2bool, default=True, help="whether to collect data")
    parser.add_argument("--manipulate_mode", type=int, default=1, help="1: right eef; 2: left eef; 3: bimanual")
    parser.add_argument('--save_video', type=str2bool, default=True, help="whether to collect save videos of camera views when storing the data")
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--task_name", type=str, required=True, help="name of the task (higher-level folder)")
    # AprilTag/board and intrinsics args
    parser.add_argument("--board_tag_size", type=float, default=0.05, help="AprilTag black square size for the board")
    parser.add_argument("--board_ids", type=str, default='0,1,2,3', help="Comma-separated AprilTag IDs for TL,TR,BR,BL")
    parser.add_argument("--tag_eye", type=str, default='left', choices=['left','right'], help="Which eye to use for AprilTag detection in stereo mode")
    parser.add_argument("--stereo_intrinsics_yaml", type=str, default='/home/robot/projects/avp_human_data/stereo_intrinsics_1.yaml', help="Path to stereo intrinsics YAML for 3D USB camera")
    parser.add_argument("--save_calibration_debug", type=str2bool, default=True, help="Whether to save calibration debug frames with drawn axes and information")
    parser.add_argument("--max_calibration_frames", type=int, default=5, help="Maximum number of frames to try for AprilTag calibration")
    parser.add_argument("--axis_length_factor", type=float, default=2.0, help="Axis length as multiple of tag size (default: 2.0)")
    parser.add_argument("--max_board_distance", type=float, default=1.5, help="Maximum board distance from camera in meters (default: 1.5)")
    # exp_name

    args = parser.parse_args()
    
    avp_teleoperator = HumanDataCollector(args)
    try:
        avp_teleoperator.run()
    finally:
        avp_teleoperator.close()

