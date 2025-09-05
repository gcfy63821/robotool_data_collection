#!/usr/bin/env python3
import tf2_ros
import rospy
import tf2_geometry_msgs
import yaml
import numpy as np
import h5py
import os
from geometry_msgs.msg import TransformStamped, PoseStamped
from std_msgs.msg import Header
from nav_msgs.msg import Path
import argparse
from collections import defaultdict
import glob


class CameraPoseVisualizer:
    def __init__(self, args):
        rospy.init_node('camera_pose_visualizer')
        
        self.args = args
        self.camera_poses = {}
        self.head_pose_trajectory = []
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Publishers
        self.head_path_pub = rospy.Publisher('/head_pose_path', Path, queue_size=10)
        
        # Head pose animation
        self.current_head_pose_index = 0
        self.head_pose_animation_speed = 0.1  # seconds between pose updates
        self.head_pose_timer = None
        
        # Board to world transform (currently identity, you can modify this later)
        self.board_to_world_transform = np.eye(4, dtype=np.float32)
        self.board_to_world_transform[0, 3] = 0.25
        
        # Rotate board frame 90 degrees around Z-axis (right-hand rule)
        # This rotates the board's X and Y axes while keeping Z the same
        angle_rad = np.pi / 2  # 90 degrees
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Rotation matrix for 90-degree Z rotation (right-hand rule)
        # This rotates X to -Y and Y to X
        self.board_to_world_transform[:3, :3] = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle,  cos_angle, 0],
            [0,          0,         1]
        ])
        
        # Load camera poses from YAML
        self.load_camera_poses()
        
        # Load head pose trajectory from data
        if self.args.data_path:
            self.load_head_pose_trajectory()
        
        # Timer for publishing TF and trajectory
        self.tf_timer = rospy.Timer(rospy.Duration(0.1), self.publish_transforms)
        
        # Timer for head pose animation
        if self.head_pose_trajectory:
            self.head_pose_timer = rospy.Timer(rospy.Duration(self.head_pose_animation_speed), self.animate_head_poses)
        
        rospy.loginfo("Camera pose visualizer initialized")
    
    def load_camera_poses(self):
        """Load camera poses from the YAML calibration file"""
        try:
            with open(self.args.calibration_yaml, 'r') as f:
                calibration_data = yaml.safe_load(f)
            
            for camera in calibration_data:
                camera_id = camera['camera_id']
                transformation = np.array(camera['transformation'])
                
                # Store the 4x4 transformation matrix
                self.camera_poses[camera_id] = transformation
                rospy.loginfo(f"Loaded camera {camera_id} pose")
                
        except Exception as e:
            rospy.logerr(f"Failed to load camera poses: {e}")
    
    def load_head_pose_trajectory(self):
        """Load head pose trajectory from saved data files"""
        try:
            # Find all episode files in the data path
            episode_files = glob.glob(os.path.join(self.args.data_path, "episode_*.hdf5"))
            
            if not episode_files:
                rospy.logwarn(f"No episode files found in {self.args.data_path}")
                return
            
            rospy.loginfo(f"Found {len(episode_files)} episode files")
            
            # If specific episode is requested, filter to that episode
            if self.args.episode is not None:
                target_episode = f"episode_{self.args.episode}.hdf5"
                episode_files = [f for f in episode_files if os.path.basename(f) == target_episode]
                if not episode_files:
                    rospy.logwarn(f"Episode {self.args.episode} not found")
                    return
                rospy.loginfo(f"Loading specific episode: {self.args.episode}")
            
            # Load head pose data from episode(s)
            for episode_file in episode_files:
                try:
                    with h5py.File(episode_file, 'r') as f:
                        rospy.loginfo(f"Loading data from {os.path.basename(episode_file)}")
                        
                        # Check if camera_poses group exists
                        if 'camera_poses' not in f:
                            rospy.logwarn(f"No camera_poses group in {os.path.basename(episode_file)}")
                            continue
                        
                        # Check if head_camera_to_init data exists
                        if 'head_camera_to_init' not in f['camera_poses']:
                            rospy.logwarn(f"No head_camera_to_init data in {os.path.basename(episode_file)}")
                            continue
                        
                        # Load head pose data - shape should be (N, 6) where N is number of frames
                        head_poses = f['camera_poses']['head_camera_to_init'][:]
                        rospy.loginfo(f"Loaded {len(head_poses)} head poses from {os.path.basename(episode_file)}")
                        
                        # Check if head_in_board_initial exists for board frame conversion
                        if 'head_in_board_initial' in f['camera_poses']:
                            head_in_board_initial = f['camera_poses']['head_in_board_initial'][:]
                            rospy.loginfo("Found head_in_board_initial transform for board frame conversion")
                            
                            # Convert each pose to board frame
                            for pose in head_poses:
                                # pose is [x, y, z, roll, pitch, yaw] - relative to initial pose in unified frame
                                # Convert to 4x4 transformation matrix
                                pose_matrix = self.rpy_to_transform_matrix(pose)
                                
                                # The head_camera_to_init represents the head pose relative to initial pose
                                # We need to convert this to board frame using head_in_board_initial
                                # 
                                # From calibrated_data_collection.py:
                                # - head_camera_to_init: head pose relative to initial pose in unified frame
                                # - head_in_board_initial: head pose relative to board frame (T_cam_in_board)
                                # 
                                # To get current head pose in board frame:
                                # T_head_in_board_current = T_head_in_board_initial @ T_head_current_relative_to_initial
                                head_in_board_current = head_in_board_initial @ pose_matrix
                                
                                self.head_pose_trajectory.append(head_in_board_current)
                        else:
                            rospy.logwarn("No head_in_board_initial found, using poses as-is (relative to initial)")
                            # Convert each pose to transformation matrix without board frame conversion
                            for pose in head_poses:
                                pose_matrix = self.rpy_to_transform_matrix(pose)
                                self.head_pose_trajectory.append(pose_matrix)
                        
                        # If we're only loading one specific episode, break after first file
                        if self.args.episode is not None:
                            break
                            
                except Exception as e:
                    rospy.logwarn(f"Failed to load episode {episode_file}: {e}")
                    continue
            
            rospy.loginfo(f"Total head poses loaded: {len(self.head_pose_trajectory)}")
            
        except Exception as e:
            rospy.logerr(f"Failed to load head pose trajectory: {e}")
    
    def animate_head_poses(self, event):
        """Animate through head poses to show movement over time"""
        if not self.head_pose_trajectory:
            return
        
        current_time = rospy.Time.now()
        
        # Get current head pose (in board frame)
        current_pose_matrix = self.head_pose_trajectory[self.current_head_pose_index]
        
        # Publish current head pose as TF under board frame
        transform_msg = TransformStamped()
        transform_msg.header.stamp = current_time
        transform_msg.header.frame_id = "board"
        transform_msg.child_frame_id = "head_camera_current"
        
        # Extract position and orientation
        position = -current_pose_matrix[:3, 3]
        rotation_matrix = current_pose_matrix[:3, :3]
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = position
        # print(T)
        quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
        
        transform_msg.transform.translation.x = position[0]
        transform_msg.transform.translation.y = position[1]
        transform_msg.transform.translation.z = position[2]
        
        transform_msg.transform.rotation.x = quaternion[0]
        transform_msg.transform.rotation.y = quaternion[1]
        transform_msg.transform.rotation.z = quaternion[2]
        transform_msg.transform.rotation.w = quaternion[3]
        
        self.tf_broadcaster.sendTransform(transform_msg)
        
        # Move to next pose (loop back to beginning)
        self.current_head_pose_index = (self.current_head_pose_index + 1) % len(self.head_pose_trajectory)
        
        # Log progress every 100 poses
        if self.current_head_pose_index % 100 == 0:
            rospy.loginfo(f"Head pose animation: {self.current_head_pose_index}/{len(self.head_pose_trajectory)}")
    
    def rpy_to_transform_matrix(self, rpy_pose):
        """Convert [x, y, z, roll, pitch, yaw] to 4x4 transformation matrix"""
        x, y, z, roll, pitch, yaw = rpy_pose
        
        # Create rotation matrix from RPY (ZYX order)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        
        return T
    
    def publish_transforms(self, event):
        """Publish all camera and head pose transforms"""
        current_time = rospy.Time.now()
        
        # Publish board to world transform (currently identity, you can modify this later)
        board_transform_msg = TransformStamped()
        board_transform_msg.header.stamp = current_time
        board_transform_msg.header.frame_id = "world"
        board_transform_msg.child_frame_id = "board"
        
        # Extract position and orientation from board_to_world_transform
        position = self.board_to_world_transform[:3, 3]
        rotation_matrix = self.board_to_world_transform[:3, :3]
        quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
        
        board_transform_msg.transform.translation.x = position[0]
        board_transform_msg.transform.translation.y = position[1]
        board_transform_msg.transform.translation.z = position[2]
        
        board_transform_msg.transform.rotation.x = quaternion[0]
        board_transform_msg.transform.rotation.y = quaternion[1]
        board_transform_msg.transform.rotation.z = quaternion[2]
        board_transform_msg.transform.rotation.w = quaternion[3]
        
        self.tf_broadcaster.sendTransform(board_transform_msg)
        
        # Publish camera poses (relative to world frame)
        for camera_id, transform_matrix in self.camera_poses.items():
            # Create transform from camera base to world
            # Note: You'll need to manually set the tf between AprilTag position and camera base
            # For now, we'll publish the camera poses relative to a world frame
            
            transform_msg = TransformStamped()
            transform_msg.header.stamp = current_time
            transform_msg.header.frame_id = "world"
            transform_msg.child_frame_id = f"camera_{camera_id}_base"
            
            # Extract position and orientation from transformation matrix
            position = transform_matrix[:3, 3]
            rotation_matrix = transform_matrix[:3, :3]
            
            # Convert rotation matrix to quaternion
            quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
            
            transform_msg.transform.translation.x = position[0]
            transform_msg.transform.translation.y = position[1]
            transform_msg.transform.translation.z = position[2]
            
            transform_msg.transform.rotation.x = quaternion[0]
            transform_msg.transform.rotation.y = quaternion[1]
            transform_msg.transform.rotation.z = quaternion[2]
            transform_msg.transform.rotation.w = quaternion[3]
            
            self.tf_broadcaster.sendTransform(transform_msg)
        
        # Publish head pose trajectory as a path (in board frame)
        if self.head_pose_trajectory:
            self.publish_head_path(current_time)
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        # Ensure R is a proper rotation matrix
        U, S, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        
        return np.array([x, y, z, w])
    
    def publish_head_path(self, current_time):
        """Publish head pose trajectory as a Path message in board frame"""
        path_msg = Path()
        path_msg.header.stamp = current_time
        path_msg.header.frame_id = "board"
        
        for i, pose_matrix in enumerate(self.head_pose_trajectory):
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = current_time
            pose_stamped.header.frame_id = "board"
            
            # Extract position and orientation
            position = pose_matrix[:3, 3]
            rotation_matrix = pose_matrix[:3, :3]
            T = np.eye(4)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = position
            T = np.linalg.inv(T)
            position = T[:3, 3]
            rotation_matrix = T[:3, :3]
            quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
            
            pose_stamped.pose.position.x = position[0]
            pose_stamped.pose.position.y = position[1]
            pose_stamped.pose.position.z = position[2]
            
            pose_stamped.pose.orientation.x = quaternion[0]
            pose_stamped.pose.orientation.y = quaternion[1]
            pose_stamped.pose.orientation.z = quaternion[2]
            pose_stamped.pose.orientation.w = quaternion[3]
            
            path_msg.poses.append(pose_stamped)
        
        self.head_path_pub.publish(path_msg)
    
    def run(self):
        """Main run loop"""
        rospy.loginfo("Camera pose visualizer running. Press Ctrl+C to stop.")
        rospy.loginfo(f"Head pose animation speed: {self.head_pose_animation_speed} seconds per pose")
        rospy.loginfo(f"Total head poses to animate: {len(self.head_pose_trajectory)}")
        rospy.loginfo("Head poses are published under 'board' frame")
        rospy.loginfo("Board frame is published under 'world' frame")
        rospy.spin()


def main():
    parser = argparse.ArgumentParser(description='Visualize camera poses and head trajectory')
    parser.add_argument('--calibration_yaml', type=str, 
                       default='/home/robot/projects/robotool/camera_ext_calibration_0821.yaml',
                       help='Path to camera calibration YAML file')
    parser.add_argument('--data_path', type=str, 
                       default='/home/robot/projects/avp_human_data/demonstrations/test/human/20250822_171052',
                       help='Path to saved data directory containing episode files')
    parser.add_argument('--episode', type=int, default=2,
                       help='Specific episode number to load (e.g., 1 for episode_1.hdf5). If not specified, loads all episodes.')
    
    args = parser.parse_args()
    
    try:
        visualizer = CameraPoseVisualizer(args)
        visualizer.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down camera pose visualizer")


if __name__ == '__main__':
    main()
