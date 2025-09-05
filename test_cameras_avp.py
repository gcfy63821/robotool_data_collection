from typing import List, Optional, Tuple

import time
import numpy as np
import trimesh
from halo import Halo
import pyrealsense2 as rs
from termcolor import colored
import cv2
import os
import shutil
import uuid
import argparse
import rospy
from std_msgs.msg import String
from datetime import datetime

import sys
sys.path.insert(0, "./")
from src.robotool.observables.camera.multi_realsense import MultiRealsense
from src.robotool.observables.observable import PointCloudAndRobotObservable
from src.robotool.io.data_writer import DataWriter

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

class SynchronizedCameraRecorder:
    def __init__(self, camera_serial, name, desired_fps=25):
        rospy.init_node('avp_camera_recorder', anonymous=True)
        
        self.camera_serial = camera_serial
        self.name = name
        self.num_cameras = len(camera_serial)
        self.desired_fps = desired_fps
        
        # ROS communication
        self.collection_status_sub = rospy.Subscriber('/avp/collection_status', String, self.collection_status_callback)
        self.ack_pub = rospy.Publisher('/avp/acknowledgment', String, queue_size=10)
        
        # Collection state
        self.collection_active = False
        self.current_episode = None
        self.waiting_for_start = True
        
        # Camera setup
        self.observable = PointCloudAndRobotObservable(
            camera_ids=camera_serial,
            camera_intrinsics=None,
            camera_transformations=None,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
        )
        
        # Data collection
        self.writer = None
        self.frame_counter = 0
        self.session_dir = None
        self.session_id = str(uuid.uuid4())[:8]
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # FPS control (from test_cameras.py)
        self.prev_time = time.time()
        
        # Send ready acknowledgment
        self.ack_pub.publish("READY")
        rospy.loginfo(f"AVP Camera Recorder initialized and ready (target FPS: {desired_fps})")
        
    def collection_status_callback(self, msg):
        """Handle collection status messages from the main system"""
        try:
            status = msg.data
            
            if status == "READY":
                rospy.loginfo("Main collection system is ready")
                
            elif status.startswith("START:"):
                episode_num = status.split(":")[1]
                rospy.loginfo(f"Starting collection for episode {episode_num}")
                self.current_episode = episode_num
                self.collection_active = True
                self.waiting_for_start = False
                self.frame_counter = 0
                
                # Create session directory
                # self.session_dir = os.path.join("/home/robot/drive/robotool/videos_0902", f"{self.session_id}_{self.name}_{episode_num}")
                self.session_dir = os.path.join(f"/home/robot/drive/robotool/videos_0902/{self.name}", f"{self.current_time}_{self.name}_{episode_num}")
                os.makedirs(self.session_dir, exist_ok=True)
                
                # Initialize writer
                self.writer = DataWriter(self.session_dir)

                # Reset observable
                self.observable.reset()
                self.writer.start(0)
                
                
                rospy.loginfo(f"Started recording to {self.session_dir}")
                
            elif status.startswith("END:"):
                episode_num = status.split(":")[1]
                rospy.loginfo(f"Ending collection for episode {episode_num}")
                self.finish_recording()
                
            elif status == "RESET":
                rospy.loginfo("Collection system reset")
                self.collection_active = False
                self.current_episode = None
                # self.finish_recording()
                
            elif status == "SHUTDOWN":
                rospy.loginfo("Collection system shutting down")
                self.finish_recording()
                rospy.signal_shutdown("Main system shutdown")
                
        except Exception as e:
            rospy.logerr(f"Error in collection status callback: {e}")
            # Send error acknowledgment
            self.ack_pub.publish("ERROR")
            
    def _normalize_latest_frames(self, obs):
        """Ensure obs contains only the latest frame per camera for each stream.
        If a stream has a time dimension (T, H, W, C) or (T, H, W), take the last frame.
        Also align lengths of imgs and depths to the minimum available.
        """
        def pick_last_frame(arr):
            # color: (H,W,C) or (T,H,W,C); depth: (H,W) or (T,H,W)
            if isinstance(arr, np.ndarray) and arr.ndim >= 4:
                return arr[-1]
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] not in (1, 3):
                # Depth could come as (T,H,W)
                return arr[-1]
            return arr

        normalized = {}
        for key, value in obs.items():
            if isinstance(value, list):
                normalized[key] = [pick_last_frame(v) for v in value]
            else:
                normalized[key] = value

        # Align imgs and depths counts
        if 'imgs' in normalized and 'depths' in normalized:
            min_len = min(len(normalized['imgs']), len(normalized['depths']))
            normalized['imgs'] = normalized['imgs'][:min_len]
            normalized['depths'] = normalized['depths'][:min_len]

        return normalized

    def finish_recording(self):
        """Finish the current recording session"""
        if self.writer is not None and self.collection_active:
            try:
                self.collection_active = False
                rospy.loginfo(f"Finishing recording for episode {self.current_episode}")
                self.writer.finish({
                    # "session_name": f"{self.session_id}_{self.current_episode}",
                    "session_name": f"{self.current_time}_{self.name}_{self.current_episode}",
                    "episode": self.current_episode
                })
                rospy.loginfo(f"Successfully finished recording episode {self.current_episode}")
            except Exception as e:
                rospy.logerr(f"Error finishing recording: {e}")
            finally:
                self.writer = None
                self.session_dir = None
                self.current_episode = None
                self.frame_counter = 0
        else:
            rospy.loginfo("No active writer to finish")
    
    def run(self):
        """Main recording loop"""
        self.observable.start()
        rospy.loginfo("Camera system started")
        
        try:
            while not rospy.is_shutdown():
                if self.collection_active and self.writer is not None:
                    try:
                        # Get observation
                        obs = self.observable.get_obs(get_points=False, depth=True, infrared=False)
                        # Normalize to ensure single latest frame per camera and aligned lengths
                        # obs = self._normalize_latest_frames(obs)
                        
                        # Skip first 50 frames for warm-up, then start recording
                        if self.frame_counter == 50:
                            rospy.loginfo("Warm-up complete, starting data collection")
                            # Send acknowledgment that recording has started
                            self.ack_pub.publish("STARTED_RECORDING")
                        
                        # if self.frame_counter < 50:
                        #     print(self.frame_counter)
                        
                        if self.frame_counter > 50 and self.frame_counter < 700:
                            # Save observation if writer is available
                            if self.writer is not None:
                                try:
                                    self.writer.save_observation(obs)
                                except Exception as e:
                                    rospy.logerr(f"Error saving observation: {e}")
                                    # Continue recording despite errors
                            else:
                                rospy.logwarn("Writer not available, skipping observation save")
                            
                            # Visualization
                            images = obs["imgs"][:8]  # first 8 images
                            rows = 2
                            cols = 4
                            
                            # Resize all images to same shape for display
                            resized_imgs = [cv2.resize(img, (100, 100)) for img in images]
                            
                            # Stack into a grid
                            row_imgs = [np.hstack(resized_imgs[i*cols:(i+1)*cols]) for i in range(rows)]
                            grid_img = np.vstack(row_imgs)
                            
                            cv2.imshow("Image Grid", grid_img)
                            
                            # Check for quit key
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord("q"):
                                rospy.loginfo("User requested quit")
                                break
                        
                        self.frame_counter += 1
                        
                        # FPS control (from test_cameras.py)
                        current_time = time.time()
                        dt = current_time - self.prev_time
                        wait_time = (1.0 / self.desired_fps) - dt
                        if wait_time > 0:
                            time.sleep(wait_time)
                        current_time = time.time()
                        print(f"\rCurrent capture rate {round(1 / (current_time - self.prev_time), 3)} fps | Frame: {self.frame_counter}", end=" ")
                        self.prev_time = current_time
                        
                    except Exception as e:
                        rospy.logerr(f"Error during recording: {e}")
                        # Continue recording despite errors
                        rospy.sleep(0.1)
                        
                else:
                    # Not actively collecting, just wait
                    rospy.sleep(0.1)
                    
        except KeyboardInterrupt:
            rospy.loginfo("Keyboard interrupt received")
        finally:
            self.finish_recording()
            self.observable.stop()
            cv2.destroyAllWindows()
            rospy.loginfo("Camera recorder shutdown complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="name of the sequence")
    parser.add_argument('--fps', type=int, default=25, help="desired FPS for recording (default: 25)")
    args = parser.parse_args()

    camera_serial = ["244222072252", 
                     "250122071059",
                     "125322060645",
                     "246422071990",
                     # "123122060454",
                     "246422071818",
                     "246422070730",
                     "250222072777",
                     "204222063088"]
    
    # Create and run the synchronized recorder
    recorder = SynchronizedCameraRecorder(camera_serial, args.name, desired_fps=args.fps)
    recorder.run()