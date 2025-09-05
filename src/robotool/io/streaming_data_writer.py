import os
import numpy as np
import h5py
import threading
import queue
import time
from typing import Dict, Any, Optional
from collections import deque
import multiprocessing as mp
from termcolor import colored


class StreamingDataWriter:
    """
    High-performance streaming data writer that writes data to disk in real-time
    instead of accumulating everything in memory.
    """
    
    def __init__(self, directory: str, buffer_size: int = 100, verbose: bool = True):
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
            print(colored(f"Started streaming writer: {file_path}", "green"))
    
    def save_observation(self, data: Dict[str, Any]) -> bool:
        """
        Save observation data asynchronously.
        Returns True if data was queued successfully, False if dropped.
        """
        if not self.is_recording or self.h5_file is None:
            return False
        
        # Debug: Print data structure on first frame
        if self.frame_counter == 0 and self.verbose:
            print(colored("Data structure analysis:", "blue"))
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"  {key}: list with {len(value)} items")
                    if value:
                        print(f"    First item type: {type(value[0])}")
                        if isinstance(value[0], np.ndarray):
                            print(f"    First item shape: {value[0].shape}, dtype: {value[0].dtype}")
                elif isinstance(value, np.ndarray):
                    print(f"  {key}: numpy array, shape: {value.shape}, dtype: {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
        
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
    
    def _background_writer(self):
        """Background thread that writes data to disk"""
        while not self.stop_event.is_set():
            try:
                # Get data from queue with timeout
                frame_idx, data = self.write_queue.get(timeout=0.1)
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
    
    def _write_frame_data(self, frame_idx: int, data: Dict[str, Any]):
        """Write a single frame of data to HDF5"""
        for key, value in data.items():
            if isinstance(value, dict):
                # Handle nested dictionaries
                for sub_key, sub_value in value.items():
                    self._write_to_dataset(f"{key}_{sub_key}", frame_idx, sub_value)
            elif isinstance(value, list):
                # Handle lists (convert to numpy array)
                if value and isinstance(value[0], np.ndarray):
                    # List of arrays - stack them
                    stacked_value = np.stack(value)
                    self._write_to_dataset(key, frame_idx, stacked_value)
                else:
                    # Regular list - convert to numpy array
                    try:
                        numpy_value = np.array(value)
                        self._write_to_dataset(key, frame_idx, numpy_value)
                    except (ValueError, TypeError):
                        # Skip if can't convert to numpy
                        continue
            else:
                self._write_to_dataset(key, frame_idx, value)
    
    def _write_to_dataset(self, key: str, frame_idx: int, value):
        """Write data to HDF5 dataset with proper chunking"""
        # Skip if value is None or not convertible
        if value is None:
            return
            
        # Convert to numpy array if not already
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value)
            except (ValueError, TypeError):
                if self.verbose:
                    print(f"Skipping non-convertible data for key '{key}': {type(value)}")
                return
        
        # Skip object arrays (dtype('O'))
        if value.dtype == object:
            if self.verbose:
                print(f"Skipping object array for key '{key}'")
            return
        
        if key not in self.datasets:
            # Create dataset on first write
            # Determine optimal chunk size
            chunk_size = min(10, max(1, self.buffer_size // 10))
            chunks = (chunk_size,) + value.shape
            
            # Create resizable dataset
            maxshape = (None,) + value.shape
            self.datasets[key] = self.h5_file.create_dataset(
                key, 
                shape=(1,) + value.shape,
                maxshape=maxshape,
                chunks=chunks,
                compression='gzip',
                compression_opts=1,
                dtype=value.dtype
            )
        
        # Resize dataset if needed
        dataset = self.datasets[key]
        if frame_idx >= dataset.shape[0]:
            new_size = max(frame_idx + 1, dataset.shape[0] * 2)
            dataset.resize((new_size,) + dataset.shape[1:])
        
        # Write data
        dataset[frame_idx] = value
    
    def finish(self, metadata: Optional[Dict] = None):
        """Finish recording and close file"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Wait for all queued data to be written
        if self.write_queue:
            self.write_queue.join()
        
        # Stop background thread
        self.stop_event.set()
        if self.write_thread and self.write_thread.is_alive():
            self.write_thread.join(timeout=2.0)
        
        # Resize datasets to actual size
        for dataset in self.datasets.values():
            dataset.resize((self.frame_counter,) + dataset.shape[1:])
        
        # Add metadata
        if metadata and self.h5_file:
            for key, value in metadata.items():
                self.h5_file.attrs[key] = value
        
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
        print(colored("Streaming DataWriter Performance Stats:", "blue"))
        print(colored(f"Total frames written: {total_frames}", "blue"))
        print(colored(f"Average write time: {avg_write_time*1000:.2f} ms", "blue"))
        print(colored(f"Max write time: {max_write_time*1000:.2f} ms", "blue"))
        print(colored(f"Dropped frames: {self.dropped_frames}", "blue"))
        print(colored(f"Drop rate: {self.dropped_frames/max(1, total_frames)*100:.2f}%", "blue"))
        print(colored("-" * 50, "blue"))


class BufferedDataWriter:
    """
    Alternative approach: Pre-allocated buffer with periodic flushing
    """
    
    def __init__(self, directory: str, buffer_size: int = 1000, flush_interval: int = 100):
        self.directory = directory
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        os.makedirs(directory, exist_ok=True)
        
        # Pre-allocated buffers
        self.buffers = {}
        self.buffer_indices = {}
        self.frame_counter = 0
        self.h5_file = None
        
    def start(self, file_index: int = 0):
        """Initialize with pre-allocated buffers"""
        file_path = os.path.join(self.directory, f"data{file_index:08d}.h5")
        self.h5_file = h5py.File(file_path, 'w')
        self.frame_counter = 0
        
    def save_observation(self, data: Dict[str, Any]):
        """Save to pre-allocated buffers"""
        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self._add_to_buffer(f"{key}_{sub_key}", sub_value)
            else:
                self._add_to_buffer(key, value)
        
        self.frame_counter += 1
        
        # Flush buffer periodically
        if self.frame_counter % self.flush_interval == 0:
            self._flush_buffers()
    
    def _add_to_buffer(self, key: str, value: np.ndarray):
        """Add data to pre-allocated buffer"""
        if key not in self.buffers:
            # Pre-allocate buffer
            self.buffers[key] = np.empty((self.buffer_size,) + value.shape, dtype=value.dtype)
            self.buffer_indices[key] = 0
        
        buffer_idx = self.buffer_indices[key] % self.buffer_size
        self.buffers[key][buffer_idx] = value
        self.buffer_indices[key] += 1
    
    def _flush_buffers(self):
        """Write buffered data to HDF5"""
        for key, buffer in self.buffers.items():
            if key not in self.h5_file:
                # Create dataset
                self.h5_file.create_dataset(
                    key,
                    data=buffer,
                    maxshape=(None,) + buffer.shape[1:],
                    chunks=True,
                    compression='gzip'
                )
            else:
                # Append to existing dataset
                dataset = self.h5_file[key]
                current_size = dataset.shape[0]
                new_size = current_size + self.buffer_size
                dataset.resize((new_size,) + dataset.shape[1:])
                dataset[current_size:] = buffer
    
    def finish(self, metadata: Optional[Dict] = None):
        """Finish and flush remaining data"""
        self._flush_buffers()
        
        if metadata:
            for key, value in metadata.items():
                self.h5_file.attrs[key] = value
        
        self.h5_file.close()
