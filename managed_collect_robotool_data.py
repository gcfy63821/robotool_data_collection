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
import threading
import queue
import h5py
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys
import json

import sys
sys.path.insert(0, "./")
from src.robotool.observables.camera.multi_realsense import MultiRealsense
from src.robotool.observables.observable import PointCloudAndRobotObservable
from src.robotool.io.data_writer import DataWriter

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


class FastBackgroundWriter:
    """
    Fast background writer that saves data as individual .npy files during recording,
    without converting to H5 format. A separate script will handle conversion later.
    """
    
    def __init__(self, directory: str, queue_size: int = 700, num_workers: int = 8):
        self.directory = directory
        self.queue_size = queue_size
        self.num_workers = num_workers
        
        # Create directories
        os.makedirs(directory, exist_ok=True)
        self.imgs_dir = os.path.join(directory, "imgs")
        self.depths_dir = os.path.join(directory, "depths")
        os.makedirs(self.imgs_dir, exist_ok=True)
        os.makedirs(self.depths_dir, exist_ok=True)
        
        # Data structures
        self.data_queue = queue.Queue(maxsize=queue_size)
        self.background_threads = []
        self.background_active = False
        self.is_recording = False
        
        # Frame counter
        self.frame_counter = 0
        
        # Thread pool for file writing
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)
        
    def start(self, file_index: int = 0):
        """Start background recording"""
        self.is_recording = True
        self.frame_counter = 0
        
        # Start background threads
        self.background_active = True
        for i in range(self.num_workers):
            thread = threading.Thread(target=self._background_worker, daemon=True, args=(i,))
            thread.start()
            self.background_threads.append(thread)
        
        print(colored(f"Started fast background writer: {self.directory}", "green"))
        print(colored(f"Queue size: {self.queue_size}, Workers: {self.num_workers}", "green"))
        print(colored(f"Images dir: {self.imgs_dir}", "green"))
        print(colored(f"Depths dir: {self.depths_dir}", "green"))
        print(colored("Note: Use convert_npy_to_h5.py script to convert to H5 format later", "blue"))
    
    def save_observation(self, obs: dict) -> bool:
        """
        Queue observation data for background processing.
        Returns True if successfully queued, False if failed.
        """
        if not self.is_recording:
            return False
        
        try:
            # Queue the data for background processing
            success = self.data_queue.put((self.frame_counter, obs), timeout=0.1)
            self.frame_counter += 1
            return success
            
        except queue.Full:
            print(colored("Data queue full, skipping frame", "yellow"))
            return False
        except Exception as e:
            print(colored(f"Error queuing observation: {e}", "red"))
            return False
    
    def _background_worker(self, worker_id: int):
        """Background worker thread for processing queued data"""
        while self.background_active:
            try:
                # Get data from queue with timeout
                frame_data = self.data_queue.get(timeout=1.0)
                if frame_data is None:  # Shutdown signal
                    break
                    
                frame_idx, obs = frame_data
                
                # Process the data (save as individual .npy files)
                self._save_frame_files(frame_idx, obs)
                
                # Mark task as done
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(colored(f"Error in background worker {worker_id}: {e}", "red"))
                if not self.data_queue.empty():
                    self.data_queue.task_done()
    
    def _save_frame_files(self, frame_idx: int, obs: dict):
        """Save frame data as individual .npy files"""
        try:
            # Save images
            if "imgs" in obs:
                imgs = obs["imgs"]
                if isinstance(imgs, list):
                    # Stack images if they're in a list
                    stacked_imgs = np.stack(imgs, axis=0)
                else:
                    stacked_imgs = imgs
                
                # Save as .npy file
                img_file = os.path.join(self.imgs_dir, f"frame_{frame_idx:08d}.npy")
                np.save(img_file, stacked_imgs)
            
            # Save depths
            if "depths" in obs:
                depths = obs["depths"]
                if isinstance(depths, list):
                    # Stack depths if they're in a list
                    stacked_depths = np.stack(depths, axis=0)
                else:
                    stacked_depths = depths
                
                # Save as .npy file
                depth_file = os.path.join(self.depths_dir, f"frame_{frame_idx:08d}.npy")
                np.save(depth_file, stacked_depths)
                
        except Exception as e:
            print(colored(f"Error saving frame {frame_idx}: {e}", "red"))
    
    def finish(self, metadata: Optional[dict] = None):
        """Finish recording and ensure all queued data is stored"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        print("Finishing fast background writer...")
        
        # Stop background threads
        self.background_active = False
        for thread in self.background_threads:
            if thread.is_alive():
                thread.join(timeout=10.0)
        
        # Wait for remaining queued data to be processed
        if not self.data_queue.empty():
            remaining_items = self.data_queue.qsize()
            print(f"Processing remaining {remaining_items} queued frames...")
            self.data_queue.join()  # Wait for all tasks to complete
            print("All queued frames processed")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Save metadata for later conversion
        self._save_metadata(metadata)
        
        print(colored("Fast background writer finished - all data stored as .npy files", "green"))
        print(colored(f"Use convert_npy_to_h5.py to convert to H5 format", "blue"))
    
    def _save_metadata(self, metadata: Optional[dict] = None):
        """Save metadata for later conversion"""
        try:
            metadata_file = os.path.join(self.directory, "metadata.json")
            import json
            
            # Count files
            img_files = glob.glob(os.path.join(self.imgs_dir, "frame_*.npy"))
            depth_files = glob.glob(os.path.join(self.depths_dir, "frame_*.npy"))
            
            metadata_dict = {
                "session_name": metadata.get("session_name", "unknown"),
                "episode": metadata.get("episode", "unknown"),
                "total_frames": metadata.get("total_frames", 0),
                "target_fps": metadata.get("target_fps", 25),
                "writer_type": "fast_background",
                "queue_size": self.queue_size,
                "num_workers": self.num_workers,
                "img_files_count": len(img_files),
                "depth_files_count": len(depth_files),
                "timestamp": time.time()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
                
            print(f"Saved metadata: {metadata_file}")
            
        except Exception as e:
            print(colored(f"Error saving metadata: {e}", "red"))


class BackgroundH5Writer:
    """
    Simple background H5 writer that queues data during recording and saves it in background.
    Based on the original DataWriter but with async processing.
    """
    
    def __init__(self, directory: str, queue_size: int = 700):
        self.directory = directory
        self.queue_size = queue_size
        
        # Create directory
        os.makedirs(directory, exist_ok=True)
        
        # Data structures
        self.data_queue = queue.Queue(maxsize=queue_size)
        self.background_thread = None
        self.background_active = False
        self.is_recording = False
        
        # H5 file
        self.h5_file = None
        self.file_path = None
        self.frame_counter = 0
        
        # Data storage
        self.imgs_data = []
        self.depths_data = []
        
    def start(self, file_index: int = 0):
        """Start background recording"""
        self.file_path = os.path.join(self.directory, f"data{file_index:08d}.h5")
        self.is_recording = True
        self.frame_counter = 0
        
        # Clear data storage
        self.imgs_data = []
        self.depths_data = []
        
        # Start background thread
        self.background_active = True
        self.background_thread = threading.Thread(target=self._background_processor, daemon=True)
        self.background_thread.start()
        
        print(colored(f"Started background H5 writer: {self.file_path}", "green"))
        print(colored(f"Queue size: {self.queue_size}", "green"))
    
    def save_observation(self, obs: dict) -> bool:
        """
        Queue observation data for background processing.
        Returns True if successfully queued, False if failed.
        """
        if not self.is_recording:
            return False
        
        try:
            # Queue the data for background processing
            success = self.data_queue.put((self.frame_counter, obs), timeout=0.1)
            self.frame_counter += 1
            return success
            
        except queue.Full:
            print(colored("Data queue full, skipping frame", "yellow"))
            return False
        except Exception as e:
            print(colored(f"Error queuing observation: {e}", "red"))
            return False
    
    def _background_processor(self):
        """Background thread for processing queued data"""
        while self.background_active:
            try:
                # Get data from queue with timeout
                frame_data = self.data_queue.get(timeout=1.0)
                if frame_data is None:  # Shutdown signal
                    break
                    
                frame_idx, obs = frame_data
                
                # Process the data (store in memory for now)
                self._process_frame(frame_idx, obs)
                
                # Mark task as done
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(colored(f"Error in background processor: {e}", "red"))
                if not self.data_queue.empty():
                    self.data_queue.task_done()
    
    def _process_frame(self, frame_idx: int, obs: dict):
        """Process frame data and store in memory"""
        try:
            # Extract image and depth data
            if "imgs" in obs:
                imgs = obs["imgs"]
                if isinstance(imgs, list):
                    # Stack images if they're in a list
                    stacked_imgs = np.stack(imgs, axis=0)
                else:
                    stacked_imgs = imgs
                self.imgs_data.append(stacked_imgs)
            
            if "depths" in obs:
                depths = obs["depths"]
                if isinstance(depths, list):
                    # Stack depths if they're in a list
                    stacked_depths = np.stack(depths, axis=0)
                else:
                    stacked_depths = depths
                self.depths_data.append(stacked_depths)
                
        except Exception as e:
            print(colored(f"Error processing frame {frame_idx}: {e}", "red"))
    
    def finish(self, metadata: Optional[dict] = None):
        """Finish recording and save all data to H5 file"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        print("Finishing background H5 writer...")
        
        # Stop background thread
        self.background_active = False
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=10.0)
        
        # Wait for remaining queued data to be processed
        if not self.data_queue.empty():
            remaining_items = self.data_queue.qsize()
            print(f"Processing remaining {remaining_items} queued frames...")
            self.data_queue.join()  # Wait for all tasks to complete
            print("All queued frames processed")
        
        # Save all data to H5 file
        self._save_to_h5(metadata)
        
        print(colored("Background H5 writer finished", "green"))
    
    def _save_to_h5(self, metadata: Optional[dict] = None):
        """Save all collected data to H5 file"""
        try:
            with h5py.File(self.file_path, 'w') as f:
                # Save image data
                if self.imgs_data:
                    imgs_array = np.stack(self.imgs_data, axis=0)
                    f.create_dataset('imgs', data=imgs_array, compression='gzip', compression_opts=1)
                    print(f"Saved {len(self.imgs_data)} image frames with shape {imgs_array.shape}")
                
                # Save depth data
                if self.depths_data:
                    depths_array = np.stack(self.depths_data, axis=0)
                    f.create_dataset('depths', data=depths_array, compression='gzip', compression_opts=1)
                    print(f"Saved {len(self.depths_data)} depth frames with shape {depths_array.shape}")
                
                # Add metadata
                if metadata:
                    for key, value in metadata.items():
                        try:
                            f.attrs[key] = value
                        except Exception as e:
                            print(f"Could not save metadata '{key}': {e}")
                
                # Add background writer metadata
                f.attrs['background_writer'] = True
                f.attrs['queue_size'] = self.queue_size
                f.attrs['total_frames'] = len(self.imgs_data)
                
        except Exception as e:
            print(colored(f"Error saving to H5 file: {e}", "red"))


class SynchronizedCameraRecorder:
    def __init__(self, camera_serial, task_name, exp_name, desired_fps=25, use_background_saving=True, queue_size=700, 
                 use_fast_writer=True, num_workers=8):
        rospy.init_node('avp_camera_recorder_optimized', anonymous=True)
        
        self.camera_serial = camera_serial
        self.task_name = task_name
        self.exp_name = exp_name
        self.num_cameras = len(camera_serial)
        self.desired_fps = desired_fps
        self.use_background_saving = use_background_saving
        self.queue_size = queue_size
        self.use_fast_writer = use_fast_writer
        self.num_workers = num_workers
        
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
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Recording tracking
        self.base_data_path = "/home/robot/drive/robotool/videos_0903"
        self.task_dir = os.path.join(self.base_data_path, self.task_name)
        self.recording_file = os.path.join(self.task_dir, "recording_tracker.json")
        self.recording_data = self._load_recording_data()
        
        # FPS control (from test_cameras.py)
        self.prev_time = time.time()
        
        # Signal handling for graceful shutdown
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Send ready acknowledgment
        self.ack_pub.publish("READY")
        if use_fast_writer:
            rospy.loginfo(f"AVP Camera Recorder initialized with fast background saving (task: {task_name}, exp: {exp_name}, target FPS: {desired_fps}, queue size: {queue_size}, workers: {num_workers})")
        elif use_background_saving:
            rospy.loginfo(f"AVP Camera Recorder initialized with background saving (task: {task_name}, exp: {exp_name}, target FPS: {desired_fps}, queue size: {queue_size})")
        else:
            rospy.loginfo(f"AVP Camera Recorder initialized with direct saving (task: {task_name}, exp: {exp_name}, target FPS: {desired_fps})")
    
    def _load_recording_data(self):
        """Load recording tracking data from file"""
        try:
            if os.path.exists(self.recording_file):
                with open(self.recording_file, 'r') as f:
                    data = json.load(f)
                    rospy.loginfo(f"Loaded recording data: {data}")
                    return data
            else:
                # Create new recording data structure
                data = {
                    "task_name": self.task_name,
                    "experiments": {},
                    "total_video_time": 0.0,
                    "last_updated": datetime.now().isoformat()
                }
                self._save_recording_data(data)
                return data
        except Exception as e:
            rospy.logwarn(f"Error loading recording data: {e}")
            return {"task_name": self.task_name, "experiments": {}, "total_video_time": 0.0, "last_updated": datetime.now().isoformat()}
    
    def _save_recording_data(self, data):
        """Save recording tracking data to file"""
        try:
            os.makedirs(self.task_dir, exist_ok=True)
            data["last_updated"] = datetime.now().isoformat()
            with open(self.recording_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            rospy.logwarn(f"Error saving recording data: {e}")
    
    def _get_next_episode_number(self):
        """Get the next episode number for the current exp name"""
        if self.exp_name not in self.recording_data["experiments"]:
            self.recording_data["experiments"][self.exp_name] = {
                "current_episode": 0,
                "total_video_time": 0.0,
                "episode_times": []
            }
        
        next_episode = self.recording_data["experiments"][self.exp_name]["current_episode"] + 1
        return next_episode
    
    def _update_recording_tracker(self, episode_num):
        """Update recording tracker with new episode"""
        if self.exp_name not in self.recording_data["experiments"]:
            self.recording_data["experiments"][self.exp_name] = {
                "current_episode": 0,
                "total_video_time": 0.0,
                "episode_times": []
            }
        
        # Update current episode number
        self.recording_data["experiments"][self.exp_name]["current_episode"] = episode_num
        
        # Save updated data
        self._save_recording_data(self.recording_data)
        
        rospy.loginfo(f"Updated recording tracker: {self.exp_name} episode {episode_num}")
    
    def _update_episode_time(self, frame_count):
        """Update episode time based on frame count"""
        if self.exp_name in self.recording_data["experiments"]:
            episode_time = frame_count / self.desired_fps  # Convert frames to seconds
            self.recording_data["experiments"][self.exp_name]["episode_times"].append(episode_time)
            
            # Update total video time for this exp
            self.recording_data["experiments"][self.exp_name]["total_video_time"] = sum(
                self.recording_data["experiments"][self.exp_name]["episode_times"]
            )
            
            # Update total video time across all experiments
            self.recording_data["total_video_time"] = sum(
                exp_data["total_video_time"] for exp_data in self.recording_data["experiments"].values()
            )
            
            # Save updated data
            self._save_recording_data(self.recording_data)
            
            rospy.loginfo(f"Updated episode time: {episode_time:.2f}s (total for {self.exp_name}: {self.recording_data['experiments'][self.exp_name]['total_video_time']:.2f}s)")
            rospy.loginfo(f"Total video time across all experiments: {self.recording_data['total_video_time']:.2f}s")
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C signal for graceful shutdown"""
        print(colored("\nCtrl+C detected. Starting graceful shutdown...", "yellow"))
        self.shutdown_requested = True
        self.finish_recording()
        rospy.signal_shutdown("User requested shutdown")
        
    def collection_status_callback(self, msg):
        """Handle collection status messages from the main system"""
        try:
            status = msg.data
            
            if status == "READY":
                rospy.loginfo("Main collection system is ready")
                
            elif status.startswith("START:"):
                # Get next episode number from recording tracker
                episode_num = self._get_next_episode_number()
                rospy.loginfo(f"Starting collection for episode {episode_num} (exp: {self.exp_name})")
                self.current_episode = episode_num
                self.collection_active = True
                self.waiting_for_start = False
                self.frame_counter = 0
                
                # Create session directory with task/exp structure
                self.session_dir = os.path.join(self.task_dir, f"{self.current_time}_{self.exp_name}_{episode_num}")
                os.makedirs(self.session_dir, exist_ok=True)
                
                # Update recording tracker
                self._update_recording_tracker(episode_num)
                
                # Initialize writer based on saving mode
                if self.use_fast_writer:
                    # Use fast background writer for multi-threaded .npy saving
                    self.writer = FastBackgroundWriter(self.session_dir, queue_size=self.queue_size, num_workers=self.num_workers)
                elif self.use_background_saving:
                    # Use background H5 writer for async saving
                    self.writer = BackgroundH5Writer(self.session_dir, queue_size=self.queue_size)
                else:
                    # Use original DataWriter for direct saving
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
                
            elif status == "SHUTDOWN":
                rospy.loginfo("Collection system shutting down")
                self.finish_recording()
                rospy.signal_shutdown("Main system shutdown")
                
        except Exception as e:
            rospy.logerr(f"Error in collection status callback: {e}")
            # Send error acknowledgment
            self.ack_pub.publish("ERROR")
            
    def finish_recording(self):
        """Finish the current recording session"""
        if self.writer is not None and self.collection_active:
            try:
                self.collection_active = False
                rospy.loginfo(f"Finishing recording for episode {self.current_episode}")
                
                # Ensure all background data is processed before finishing
                if (self.use_background_saving or self.use_fast_writer) and hasattr(self.writer, 'data_queue'):
                    # Wait for all queued data to be processed
                    remaining_items = self.writer.data_queue.qsize()
                    if remaining_items > 0:
                        rospy.loginfo(f"Processing remaining {remaining_items} queued frames...")
                        self.writer.data_queue.join()  # Wait for all tasks to complete
                        rospy.loginfo("All queued frames processed")
                
                # Update episode time in recording tracker
                self._update_episode_time(self.frame_counter)
                
                # Finish the writer
                if self.use_fast_writer:
                    self.writer.finish({
                        "session_name": f"{self.current_time}_{self.exp_name}_{self.current_episode}",
                        "episode": self.current_episode,
                        "total_frames": self.frame_counter,
                        "target_fps": self.desired_fps,
                        "writer_type": "fast_background",
                        "task_name": self.task_name,
                        "exp_name": self.exp_name
                    })
                elif self.use_background_saving:
                    self.writer.finish({
                        "session_name": f"{self.current_time}_{self.exp_name}_{self.current_episode}",
                        "episode": self.current_episode,
                        "total_frames": self.frame_counter,
                        "target_fps": self.desired_fps,
                        "task_name": self.task_name,
                        "exp_name": self.exp_name
                    })
                else:
                    self.writer.finish({
                        "session_name": f"{self.current_time}_{self.exp_name}_{self.current_episode}",
                        "episode": self.current_episode,
                        "task_name": self.task_name,
                        "exp_name": self.exp_name
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
            while not rospy.is_shutdown() and not self.shutdown_requested:
                if self.collection_active and self.writer is not None:
                    try:
                        # Get observation
                        obs = self.observable.get_obs(get_points=False, depth=True, infrared=False)
                        
                        # Skip first 50 frames for warm-up, then start recording
                        if self.frame_counter == 50:
                            rospy.loginfo("Warm-up complete, starting data collection")
                            # Send acknowledgment that recording has started
                            self.ack_pub.publish("STARTED_RECORDING")
                        
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
    parser.add_argument('--task_name', type=str, required=True, help="name of the task (higher-level folder)")
    parser.add_argument('--exp_name', type=str, required=True, help="name of the experiment (under task folder)")
    parser.add_argument('--fps', type=int, default=25, help="desired FPS for recording (default: 25)")
    parser.add_argument('--background', action='store_true', help="use background saving (memory-based)")
    parser.add_argument('--direct', action='store_true', help="use direct saving instead of background")
    parser.add_argument('--fast', action='store_true', default=True, help="use fast multi-threaded .npy saving (default: True)")
    parser.add_argument('--no_fast', action='store_true', help="disable fast saving")
    parser.add_argument('--queue_size', type=int, default=700, help="queue size for background saving (default: 700)")
    parser.add_argument('--num_workers', type=int, default=8, help="number of worker threads for fast saving (default: 8)")
    args = parser.parse_args()

    camera_serial = ["244222072252", 
                     "250122071059",
                     "125322060645",
                     "246422071990",
                     "246422071818",
                     "246422070730",
                     "250222072777",
                     "204222063088"]
    
    # Determine saving mode
    use_background_saving = args.background and not args.direct
    use_fast_writer = args.fast and not args.no_fast
    
    print(f"Camera configuration:")
    print(f"  Task name: {args.task_name}")
    print(f"  Experiment name: {args.exp_name}")
    print(f"  Total cameras: {len(camera_serial)}")
    if use_fast_writer:
        print(f"  Saving mode: Fast Multi-threaded (.npy -> H5)")
        print(f"  Queue size: {args.queue_size}")
        print(f"  Workers: {args.num_workers}")
    elif use_background_saving:
        print(f"  Saving mode: Background (Memory -> H5)")
        print(f"  Queue size: {args.queue_size}")
    else:
        print(f"  Saving mode: Direct (Original)")
    print(f"  Target FPS: {args.fps}")
    
    # Create and run the synchronized recorder
    recorder = SynchronizedCameraRecorder(
        camera_serial, 
        args.task_name,
        args.exp_name,
        desired_fps=args.fps,
        use_background_saving=use_background_saving,
        queue_size=args.queue_size,
        use_fast_writer=use_fast_writer,
        num_workers=args.num_workers
    )
    recorder.run()
