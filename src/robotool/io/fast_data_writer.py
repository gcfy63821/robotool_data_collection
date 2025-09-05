import os
import numpy as np
import h5py
import time
from typing import Dict, Any, Optional
# from termcolor import colored
def colored(text, color=None):
    return text


class FastDataWriter:
    """
    Ultra-fast data writer with mixed saving modes capability.
    Supports both real-time saving and post-processing saving for different cameras.
    Can save some cameras in real-time while buffering others for post-processing.
    Optimized for real-time camera data recording at 25+ FPS.
    """
    
    def __init__(self, directory: str, verbose: bool = True, deferred_write: bool = True, 
                 realtime_cameras: list = None, postprocess_cameras: list = None):
        self.directory = directory
        self.verbose = verbose
        self.deferred_write = deferred_write
        
        # Camera saving mode configuration
        self.realtime_cameras = realtime_cameras or []  # List of camera indices for real-time saving
        self.postprocess_cameras = postprocess_cameras or []  # List of camera indices for post-processing
        self.mixed_mode = bool(realtime_cameras or postprocess_cameras)  # True if using mixed modes
        
        # Create directory
        os.makedirs(directory, exist_ok=True)
        
        # Data structures
        self.h5_file = None
        self.datasets = {}
        self.frame_counter = 0
        self.is_recording = False
        
        # Memory buffer for deferred writing (post-processing cameras)
        self.memory_buffer = {}
        self.buffer_shapes = {}
        self.buffer_types = {}
        
        # Real-time data storage (for real-time cameras)
        self.realtime_buffer = {}
        self.realtime_shapes = {}
        self.realtime_types = {}
        
        # Performance tracking
        self.write_times = []
        self.save_times = []
        self.data_shapes = {}
        self.data_types = {}
        
    def start(self, file_index: int = 0):
        """Initialize and start recording"""
        self.file_path = os.path.join(self.directory, f"data{file_index:08d}.h5")
        self.is_recording = True
        self.frame_counter = 0
        
        # Clear memory buffers
        self.memory_buffer = {}
        self.buffer_shapes = {}
        self.buffer_types = {}
        self.realtime_buffer = {}
        self.realtime_shapes = {}
        self.realtime_types = {}
        
        if self.verbose:
            if self.mixed_mode:
                rt_cams = len(self.realtime_cameras)
                pp_cams = len(self.postprocess_cameras)
                print(colored(f"Started mixed-mode fast writer: {self.file_path}", "green"))
                print(colored(f"  Real-time cameras: {rt_cams} (indices: {self.realtime_cameras})", "green"))
                print(colored(f"  Post-process cameras: {pp_cams} (indices: {self.postprocess_cameras})", "green"))
            else:
                mode = "deferred" if self.deferred_write else "direct"
                print(colored(f"Started fast writer ({mode}): {self.file_path}", "green"))
    
    def save_observation(self, data: Dict[str, Any]) -> bool:
        """
        Save observation data with mixed modes support.
        Real-time cameras are saved immediately, post-process cameras are buffered.
        Returns True if successful, False if failed.
        """
        if not self.is_recording:
            return False
        
        # Debug: Print data structure on first frame
        if self.frame_counter == 0 and self.verbose:
            self._analyze_data_structure(data)
        
        try:
            start_time = time.time()
            
            if self.mixed_mode:
                # Handle mixed mode: split data by camera
                self._save_mixed_mode_data(data)
            else:
                # Handle traditional mode
                if self.deferred_write:
                    # Store in memory buffer for later writing
                    self._store_frame_in_buffer(data)
                else:
                    # Write directly to HDF5
                    if self.h5_file is None:
                        self.h5_file = h5py.File(self.file_path, 'w')
                    self._write_frame_data(data)
            
            save_time = time.time() - start_time
            self.save_times.append(save_time)
            
            self.frame_counter += 1
            return True
            
        except Exception as e:
            if self.verbose:
                print(colored(f"Error saving observation: {e}", "red"))
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
            elif isinstance(value, np.ndarray):
                print(f"  Type: numpy array")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
            else:
                print(f"  Type: {type(value)}")
            print()
    
    def _save_mixed_mode_data(self, data: Dict[str, Any]):
        """Handle mixed mode data saving - split by camera type"""
        for key, value in data.items():
            if key in ["imgs", "depths"] and isinstance(value, list):
                # Split camera data by type
                realtime_data = []
                postprocess_data = []
                
                for i, item in enumerate(value):
                    if i in self.realtime_cameras:
                        realtime_data.append(item)
                    elif i in self.postprocess_cameras:
                        postprocess_data.append(item)
                
                # Save real-time data immediately
                if realtime_data:
                    realtime_key = f"{key}_realtime"
                    if key == "imgs":
                        stacked_realtime = np.stack(realtime_data, axis=0)
                    else:  # depths
                        stacked_realtime = np.stack(realtime_data, axis=0)
                    
                    # Write real-time data directly to HDF5
                    if self.h5_file is None:
                        self.h5_file = h5py.File(self.file_path, 'w')
                    self._write_array_data(realtime_key, stacked_realtime)
                
                # Buffer post-process data
                if postprocess_data:
                    postprocess_key = f"{key}_postprocess"
                    if key == "imgs":
                        stacked_postprocess = np.stack(postprocess_data, axis=0)
                    else:  # depths
                        stacked_postprocess = np.stack(postprocess_data, axis=0)
                    
                    self._store_array_in_buffer(postprocess_key, stacked_postprocess)
            else:
                # Handle non-camera data (save to both modes or default to post-process)
                if self.deferred_write:
                    self._store_array_in_buffer(key, value)
                else:
                    if self.h5_file is None:
                        self.h5_file = h5py.File(self.file_path, 'w')
                    self._write_array_data(key, value)
    
    def _store_frame_in_buffer(self, data: Dict[str, Any]):
        """Store frame data in memory buffer for deferred writing"""
        for key, value in data.items():
            if key == "imgs" and isinstance(value, list):
                # Handle image data specifically
                stacked_images = np.stack(value, axis=0)
                self._store_array_in_buffer(key, stacked_images)
            elif key == "depths" and isinstance(value, list):
                # Handle depth data specifically
                stacked_depths = np.stack(value, axis=0)
                self._store_array_in_buffer(key, stacked_depths)
            elif isinstance(value, list):
                # Handle other lists
                if all(isinstance(item, np.ndarray) for item in value):
                    stacked_data = np.stack(value, axis=0)
                    self._store_array_in_buffer(key, stacked_data)
            elif isinstance(value, np.ndarray):
                # Handle numpy arrays
                self._store_array_in_buffer(key, value)
    
    def _store_array_in_buffer(self, key: str, value: np.ndarray):
        """Store numpy array in memory buffer"""
        if value is None or value.dtype == object:
            return
        
        if key not in self.memory_buffer:
            # Initialize buffer for this key
            self.buffer_shapes[key] = value.shape
            self.buffer_types[key] = value.dtype
            # Pre-allocate buffer with some extra space
            initial_size = max(100, 200)  # Start with 200 frames
            self.memory_buffer[key] = np.empty(
                (initial_size,) + value.shape, 
                dtype=value.dtype
            )
            self.memory_buffer[key][0] = value
            self.memory_buffer[key] = self.memory_buffer[key][:1]  # Resize to actual size
        else:
            # Add to existing buffer
            buffer = self.memory_buffer[key]
            if self.frame_counter >= buffer.shape[0]:
                # Resize buffer (double the size)
                new_size = buffer.shape[0] * 2
                new_buffer = np.empty(
                    (new_size,) + self.buffer_shapes[key],
                    dtype=self.buffer_types[key]
                )
                new_buffer[:len(buffer)] = buffer
                self.memory_buffer[key] = new_buffer
            
            # Add the new value
            self.memory_buffer[key][self.frame_counter] = value
    
    def _write_frame_data(self, data: Dict[str, Any]):
        """Write a single frame of data to HDF5"""
        for key, value in data.items():
            if key == "imgs" and isinstance(value, list):
                # Handle image data specifically
                self._write_image_data(key, value)
            elif key == "depths" and isinstance(value, list):
                # Handle depth data specifically
                self._write_depth_data(key, value)
            elif isinstance(value, list):
                # Handle other lists
                self._write_list_data(key, value)
            elif isinstance(value, np.ndarray):
                # Handle numpy arrays
                self._write_array_data(key, value)
    
    def _write_image_data(self, key: str, images: list):
        """Write image data (list of RGB arrays)"""
        if not images:
            return
        
        # Stack images into a single array: (num_cameras, height, width, channels)
        stacked_images = np.stack(images, axis=0)
        self._write_array_data(key, stacked_images)
    
    def _write_depth_data(self, key: str, depths: list):
        """Write depth data (list of depth arrays)"""
        if not depths:
            return
        
        # Stack depth maps into a single array: (num_cameras, height, width)
        stacked_depths = np.stack(depths, axis=0)
        self._write_array_data(key, stacked_depths)
    
    def _write_list_data(self, key: str, data_list: list):
        """Write generic list data"""
        if not data_list:
            return
        
        # Try to convert to numpy array
        if all(isinstance(item, np.ndarray) for item in data_list):
            # All items are numpy arrays - stack them
            stacked_data = np.stack(data_list, axis=0)
            self._write_array_data(key, stacked_data)
    
    def _write_array_data(self, key: str, value: np.ndarray):
        """Write numpy array data to HDF5 dataset"""
        if value is None or value.dtype == object:
            return
        
        if key not in self.datasets:
            # Create dataset on first write
            self._create_dataset(key, value)
        
        # Resize dataset if needed
        dataset = self.datasets[key]
        if self.frame_counter >= dataset.shape[0]:
            new_size = max(self.frame_counter + 1, dataset.shape[0] * 2)
            dataset.resize((new_size,) + dataset.shape[1:])
        
        # Write data
        dataset[self.frame_counter] = value
    
    def _create_dataset(self, key: str, value: np.ndarray):
        """Create HDF5 dataset with optimal settings for speed"""
        # Create resizable dataset with minimal overhead
        maxshape = (None,) + value.shape
        
        # Use no compression for maximum write speed
        self.datasets[key] = self.h5_file.create_dataset(
            key, 
            shape=(1,) + value.shape,
            maxshape=maxshape,
            chunks=True,  # Let HDF5 choose optimal chunks
            compression=None,  # No compression for speed
            dtype=value.dtype
        )
        
        if self.verbose:
            print(f"Created dataset '{key}' with shape {value.shape}, dtype {value.dtype}")
    
    def finish(self, metadata: Optional[Dict] = None):
        """Finish recording and write data to disk"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.verbose:
            print("Finishing recording and writing data to disk...")
        
        if self.mixed_mode:
            # Handle mixed mode finishing
            self._finish_mixed_mode(metadata)
        else:
            # Handle traditional mode finishing
            if self.deferred_write:
                # Write buffered data to HDF5
                self._write_buffered_data_to_disk(metadata)
            else:
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
    
    def _finish_mixed_mode(self, metadata: Optional[Dict] = None):
        """Finish mixed mode recording - handle both real-time and post-process data"""
        if self.verbose:
            print("Finishing mixed-mode recording...")
        
        # Ensure HDF5 file is open
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'w')
        
        # Resize real-time datasets to actual size
        for dataset in self.datasets.values():
            if dataset.name.endswith('_realtime'):
                dataset.resize((self.frame_counter,) + dataset.shape[1:])
        
        # Write post-process buffered data
        if self.memory_buffer:
            if self.verbose:
                print(f"Writing {self.frame_counter} frames of post-process data from memory buffer...")
            
            for key, buffer in self.memory_buffer.items():
                # Trim buffer to actual size
                actual_buffer = buffer[:self.frame_counter]
                
                # Create dataset and write data
                self.h5_file.create_dataset(
                    key,
                    data=actual_buffer,
                    compression='gzip',
                    compression_opts=1
                )
                
                if self.verbose:
                    print(f"Written post-process dataset '{key}': shape={actual_buffer.shape}, dtype={actual_buffer.dtype}")
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                try:
                    self.h5_file.attrs[key] = value
                except Exception as e:
                    if self.verbose:
                        print(f"Could not save metadata '{key}': {e}")
        
        # Add mixed-mode specific metadata
        if self.h5_file:
            self.h5_file.attrs['mixed_mode'] = True
            self.h5_file.attrs['realtime_cameras'] = self.realtime_cameras
            self.h5_file.attrs['postprocess_cameras'] = self.postprocess_cameras
        
        # Close file
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None
        
        if self.verbose:
            print("Mixed-mode recording finished successfully")
    
    def _write_buffered_data_to_disk(self, metadata: Optional[Dict] = None):
        """Write all buffered data to HDF5 file"""
        if self.verbose:
            print(f"Writing {self.frame_counter} frames from memory buffer to disk...")
        
        start_time = time.time()
        
        # Create HDF5 file
        self.h5_file = h5py.File(self.file_path, 'w')
        
        # Write all buffered data
        for key, buffer in self.memory_buffer.items():
            # Trim buffer to actual size
            actual_buffer = buffer[:self.frame_counter]
            
            # Create dataset and write data
            self.h5_file.create_dataset(
                key,
                data=actual_buffer,
                compression='gzip',
                compression_opts=1
            )
            
            if self.verbose:
                print(f"Written dataset '{key}': shape={actual_buffer.shape}, dtype={actual_buffer.dtype}")
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                try:
                    self.h5_file.attrs[key] = value
                except Exception as e:
                    if self.verbose:
                        print(f"Could not save metadata '{key}': {e}")
        
        # Close file
        self.h5_file.close()
        self.h5_file = None
        
        write_time = time.time() - start_time
        if self.verbose:
            print(f"Deferred write completed in {write_time:.2f} seconds")
            print(f"Write speed: {self.frame_counter/write_time:.1f} frames/second")
    
    def _print_performance_stats(self):
        """Print performance statistics"""
        if not self.save_times:
            return
        
        avg_save_time = np.mean(self.save_times) * 1000
        max_save_time = np.max(self.save_times) * 1000
        min_save_time = np.min(self.save_times) * 1000
        total_frames = self.frame_counter
        
        print(colored("-" * 50, "blue"))
        print(colored("Fast DataWriter Performance Stats:", "blue"))
        print(colored(f"Total frames processed: {total_frames}", "blue"))
        print(colored(f"Average save time: {avg_save_time:.2f} ms", "blue"))
        print(colored(f"Min save time: {min_save_time:.2f} ms", "blue"))
        print(colored(f"Max save time: {max_save_time:.2f} ms", "blue"))
        print(colored(f"Estimated FPS: {1000/avg_save_time:.1f}", "blue"))
        if self.mixed_mode:
            print(colored(f"Mode: Mixed (Real-time: {len(self.realtime_cameras)}, Post-process: {len(self.postprocess_cameras)})", "blue"))
            print(colored(f"Real-time datasets: {len([d for d in self.datasets.keys() if d.endswith('_realtime')])}", "blue"))
            print(colored(f"Post-process datasets: {len(self.memory_buffer)}", "blue"))
        else:
            print(colored(f"Mode: {'Deferred' if self.deferred_write else 'Direct'}", "blue"))
            print(colored(f"Datasets created: {len(self.memory_buffer) if self.deferred_write else len(self.datasets)}", "blue"))
        print(colored("-" * 50, "blue"))


# Utility function to create fast writer
def create_fast_writer(directory: str, **kwargs) -> FastDataWriter:
    """Create a fast data writer with the same interface as the original"""
    return FastDataWriter(directory, **kwargs)
