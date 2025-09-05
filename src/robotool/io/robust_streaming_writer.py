import os
import numpy as np
import h5py
import threading
import queue
import time
from typing import Dict, Any, Optional, List
from collections import deque
import multiprocessing as mp
from termcolor import colored


class RobustStreamingDataWriter:
    """
    Robust streaming data writer specifically designed for camera observation data.
    Handles the specific data structure from MultiRealsense cameras.
    """
    
    def __init__(self, directory: str, buffer_size: int = 200, verbose: bool = True):
        self.directory = directory
        self.buffer_size = buffer_size
        self.verbose = verbose
        
        # Create directory
        os.makedirs(directory, exist_ok=True)
        
        # Data structures
        self.h5_file = None
        self.datasets = {}
        self.frame_counter = 0
        self.is_recording = False
        
        # Threading for async writing
        self.write_queue = queue.Queue(maxsize=buffer_size)
        self.write_thread = None
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.write_times = deque(maxlen=100)
        self.dropped_frames = 0
        
        # Data structure tracking
        self.data_shapes = {}
        self.data_types = {}
        
    def start(self, file_index: int = 0):
        """Initialize HDF5 file and start recording"""
        file_path = os.path.join(self.directory, f"data{file_index:08d}.h5")
        self.h5_file = h5py.File(file_path, 'w')
        self.is_recording = True
        self.frame_counter = 0
        
        # Start background writer thread
        self.write_thread = threading.Thread(target=self._background_writer, daemon=True)
        self.write_thread.start()
        
        if self.verbose:
            print(colored(f"Started robust streaming writer: {file_path}", "green"))
    
    def save_observation(self, data: Dict[str, Any]) -> bool:
        """
        Save observation data asynchronously.
        Returns True if data was queued successfully, False if dropped.
        """
        if not self.is_recording or self.h5_file is None:
            return False
        
        # Debug: Print data structure on first frame
        if self.frame_counter == 0 and self.verbose:
            self._analyze_data_structure(data)
        
        try:
            # Try to put data in queue (non-blocking)
            self.write_queue.put_nowait((self.frame_counter, data))
            self.frame_counter += 1
            return True
        except queue.Full:
            # Drop frame if buffer is full
            self.dropped_frames += 1
            if self.verbose and self.dropped_frames % 10 == 0:
                print(colored(f"Warning: Dropped {self.dropped_frames} frames", "yellow"))
            return False
    
    def _analyze_data_structure(self, data: Dict[str, Any]):
        """Analyze and print the data structure for debugging"""
        print(colored("=" * 50, "blue"))
        print(colored("DATA STRUCTURE ANALYSIS", "blue"))
        print(colored("=" * 50, "blue"))
        
        for key, value in data.items():
            print(f"Key: '{key}'")
            if isinstance(value, list):
                print(f"  Type: list with {len(value)} items")
                if value:
                    first_item = value[0]
                    print(f"  First item type: {type(first_item)}")
                    if isinstance(first_item, np.ndarray):
                        print(f"  First item shape: {first_item.shape}")
                        print(f"  First item dtype: {first_item.dtype}")
                    elif isinstance(first_item, dict):
                        print(f"  First item dict keys: {list(first_item.keys())}")
            elif isinstance(value, np.ndarray):
                print(f"  Type: numpy array")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
            else:
                print(f"  Type: {type(value)}")
                print(f"  Value: {value}")
            print()
    
    def _background_writer(self):
        """Background thread that writes data to disk"""
        if self.verbose:
            print("Background writer thread started")
        
        while not self.stop_event.is_set():
            try:
                # Get data from queue with timeout
                frame_idx, data = self.write_queue.get(timeout=0.1)
                if self.verbose and frame_idx % 10 == 0:
                    print(f"Background writer processing frame {frame_idx}")
                
                start_time = time.time()
                
                # Write data to HDF5
                self._write_frame_data(frame_idx, data)
                
                # Track performance
                write_time = time.time() - start_time
                self.write_times.append(write_time)
                
                self.write_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.verbose:
                    print(colored(f"Error in background writer: {e}", "red"))
                    import traceback
                    traceback.print_exc()
        
        if self.verbose:
            print("Background writer thread stopped")
    
    def _write_frame_data(self, frame_idx: int, data: Dict[str, Any]):
        """Write a single frame of data to HDF5 with robust handling"""
        for key, value in data.items():
            if key == "imgs" and isinstance(value, list):
                # Handle image data specifically
                self._write_image_data(key, frame_idx, value)
            elif key == "depths" and isinstance(value, list):
                # Handle depth data specifically
                self._write_depth_data(key, frame_idx, value)
            elif isinstance(value, list):
                # Handle other lists
                self._write_list_data(key, frame_idx, value)
            elif isinstance(value, np.ndarray):
                # Handle numpy arrays
                self._write_array_data(key, frame_idx, value)
            else:
                # Skip other types
                if self.verbose and frame_idx == 0:
                    print(f"Skipping non-handled data type for key '{key}': {type(value)}")
    
    def _write_image_data(self, key: str, frame_idx: int, images: List[np.ndarray]):
        """Write image data (list of RGB arrays)"""
        if not images:
            return
        
        # Stack images into a single array: (num_cameras, height, width, channels)
        try:
            stacked_images = np.stack(images, axis=0)
            self._write_array_data(key, frame_idx, stacked_images)
        except Exception as e:
            if self.verbose:
                print(f"Error stacking images: {e}")
    
    def _write_depth_data(self, key: str, frame_idx: int, depths: List[np.ndarray]):
        """Write depth data (list of depth arrays)"""
        if not depths:
            return
        
        # Stack depth maps into a single array: (num_cameras, height, width)
        try:
            stacked_depths = np.stack(depths, axis=0)
            self._write_array_data(key, frame_idx, stacked_depths)
        except Exception as e:
            if self.verbose:
                print(f"Error stacking depths: {e}")
    
    def _write_list_data(self, key: str, frame_idx: int, data_list: List):
        """Write generic list data"""
        if not data_list:
            return
        
        # Try to convert to numpy array
        try:
            if all(isinstance(item, np.ndarray) for item in data_list):
                # All items are numpy arrays - stack them
                stacked_data = np.stack(data_list, axis=0)
                self._write_array_data(key, frame_idx, stacked_data)
            else:
                # Mixed types - convert to object array (skip for HDF5)
                if self.verbose and frame_idx == 0:
                    print(f"Skipping mixed-type list for key '{key}'")
        except Exception as e:
            if self.verbose:
                print(f"Error processing list data for key '{key}': {e}")
    
    def _write_array_data(self, key: str, frame_idx: int, value: np.ndarray):
        """Write numpy array data to HDF5 dataset"""
        # Skip if value is None or not convertible
        if value is None:
            return
        
        # Ensure it's a numpy array
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value)
            except (ValueError, TypeError):
                if self.verbose:
                    print(f"Skipping non-convertible data for key '{key}': {type(value)}")
                return
        
        # Skip object arrays (dtype('O'))
        if value.dtype == object:
            if self.verbose and frame_idx == 0:
                print(f"Skipping object array for key '{key}'")
            return
        
        # Ensure data type is HDF5 compatible
        if value.dtype.kind in ['U', 'S']:  # Unicode or byte strings
            if self.verbose and frame_idx == 0:
                print(f"Skipping string array for key '{key}'")
            return
        
        if key not in self.datasets:
            # Create dataset on first write
            self._create_dataset(key, value)
        
        # Resize dataset if needed
        dataset = self.datasets[key]
        if frame_idx >= dataset.shape[0]:
            new_size = max(frame_idx + 1, dataset.shape[0] * 2)
            dataset.resize((new_size,) + dataset.shape[1:])
        
        # Write data
        try:
            dataset[frame_idx] = value
        except Exception as e:
            if self.verbose:
                print(f"Error writing data for key '{key}': {e}")
    
    def _create_dataset(self, key: str, value: np.ndarray):
        """Create HDF5 dataset with optimal settings for performance"""
        # Determine optimal chunk size - larger chunks for better performance
        chunk_size = min(50, max(10, self.buffer_size // 4))
        chunks = (chunk_size,) + value.shape
        
        # Create resizable dataset
        maxshape = (None,) + value.shape
        
        # Use lighter compression for better write performance
        if value.dtype in [np.uint8, np.uint16]:
            # Image data - use lighter compression for speed
            compression = 'gzip'
            compression_opts = 1  # Reduced from 6 to 1 for speed
        else:
            # Other data - no compression for maximum speed
            compression = None
            compression_opts = None
        
        self.datasets[key] = self.h5_file.create_dataset(
            key, 
            shape=(1,) + value.shape,
            maxshape=maxshape,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
            dtype=value.dtype,
            shuffle=False  # Disable shuffle for speed
        )
        
        if self.verbose:
            print(f"Created dataset '{key}' with shape {value.shape}, dtype {value.dtype}, compression={compression}")
    
    def finish(self, metadata: Optional[Dict] = None):
        """Finish recording and close file"""
        if not self.is_recording:
            return
        
        if self.verbose:
            print("Finishing streaming writer...")
        
        self.is_recording = False
        
        # Wait for all queued data to be written
        if self.write_queue:
            if self.verbose:
                print(f"Waiting for {self.write_queue.qsize()} queued items to be processed...")
            self.write_queue.join()
            if self.verbose:
                print("All queued items processed")
        
        # Stop background thread
        self.stop_event.set()
        if self.write_thread and self.write_thread.is_alive():
            if self.verbose:
                print("Stopping background writer thread...")
            self.write_thread.join(timeout=2.0)
            if self.verbose:
                print("Background writer thread stopped")
        
        # Resize datasets to actual size
        for dataset in self.datasets.values():
            dataset.resize((self.frame_counter,) + dataset.shape[1:])
        
        # Add metadata
        if metadata and self.h5_file:
            for key, value in metadata.items():
                try:
                    self.h5_file.attrs[key] = value
                except Exception as e:
                    if self.verbose:
                        print(f"Could not save metadata '{key}': {e}")
        
        # Close file
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None
        
        # Print performance stats
        if self.verbose:
            self._print_performance_stats()
    
    def _print_performance_stats(self):
        """Print performance statistics"""
        if not self.write_times:
            return
        
        avg_write_time = np.mean(self.write_times)
        max_write_time = np.max(self.write_times)
        total_frames = self.frame_counter
        
        print(colored("-" * 50, "blue"))
        print(colored("Robust Streaming DataWriter Performance Stats:", "blue"))
        print(colored(f"Total frames written: {total_frames}", "blue"))
        print(colored(f"Average write time: {avg_write_time*1000:.2f} ms", "blue"))
        print(colored(f"Max write time: {max_write_time*1000:.2f} ms", "blue"))
        print(colored(f"Dropped frames: {self.dropped_frames}", "blue"))
        print(colored(f"Drop rate: {self.dropped_frames/max(1, total_frames)*100:.2f}%", "blue"))
        print(colored(f"Datasets created: {len(self.datasets)}", "blue"))
        print(colored("-" * 50, "blue"))
