import os
import time
import numpy as np
import h5py
from typing import Dict, Any, Optional, List
from termcolor import colored
from halo import Halo


def colored(text, color=None):
    return text


class OptimizedDataWriter:
    """
    Optimized data writer based on the original DataWriter but with significant performance improvements.
    
    Key optimizations:
    1. Direct HDF5 writing instead of Zarr intermediate step
    2. Pre-allocated datasets with resizable dimensions
    3. Efficient chunking strategy
    4. Reduced memory allocations
    5. Faster compression with better algorithms
    6. Batch processing for better I/O performance
    """
    
    def __init__(self, directory: str, verbose: bool = True, chunk_size: int = 25, 
                 compression: str = 'lzf', compression_opts: int = None):
        self.directory = directory
        self.verbose = verbose
        self.chunk_size = chunk_size
        self.compression = compression
        self.compression_opts = compression_opts
        
        # Create directory
        os.makedirs(directory, exist_ok=True)
        
        # Data structures
        self.h5_file = None
        self.datasets = {}
        self.frame_counter = 0
        self.is_recording = False
        
        # Performance tracking
        self.write_times = []
        self.save_times = []
        self.data_shapes = {}
        self.data_types = {}
        
        # Batch processing
        self.batch_buffer = {}
        self.batch_size = 0
        
    def start(self, file_index: int = 0):
        """Initialize and start recording"""
        self.file_path = os.path.join(self.directory, f"data{file_index:08d}.h5")
        self.is_recording = True
        self.frame_counter = 0
        self.batch_buffer = {}
        self.batch_size = 0
        
        if self.verbose:
            print(colored(f"Started optimized writer: {self.file_path}", "green"))
            print(colored(f"  Chunk size: {self.chunk_size}", "green"))
            print(colored(f"  Compression: {self.compression}", "green"))
    
    def save_observation(self, data: Dict[str, Any]) -> bool:
        """
        Save observation data with optimized performance.
        Returns True if successful, False if failed.
        """
        if not self.is_recording:
            return False
        
        # Debug: Print data structure on first frame
        if self.frame_counter == 0 and self.verbose:
            self._analyze_data_structure(data)
        
        try:
            start_time = time.time()
            
            # Process data and add to batch buffer
            self._process_observation_data(data)
            
            save_time = time.time() - start_time
            self.save_times.append(save_time)
            
            self.frame_counter += 1
            self.batch_size += 1
            
            # Write batch if it reaches chunk size
            if self.batch_size >= self.chunk_size:
                self._write_batch()
                self.batch_size = 0
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(colored(f"Error saving observation: {e}", "red"))
            return False
    
    def _analyze_data_structure(self, data: Dict[str, Any]):
        """Analyze and print the data structure for debugging"""
        print(colored("=" * 50, "blue"))
        print(colored("OPTIMIZED DATA WRITER - DATA STRUCTURE ANALYSIS", "blue"))
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
    
    def _process_observation_data(self, data: Dict[str, Any]):
        """Process observation data and add to batch buffer"""
        for key, value in data.items():
            if key == "imgs" and isinstance(value, list):
                # Handle image data specifically - stack into single array
                stacked_images = np.stack(value, axis=0)
                self._add_to_batch_buffer(key, stacked_images)
            elif key == "depths" and isinstance(value, list):
                # Handle depth data specifically - stack into single array
                stacked_depths = np.stack(value, axis=0)
                self._add_to_batch_buffer(key, stacked_depths)
            elif isinstance(value, list):
                # Handle other lists
                if all(isinstance(item, np.ndarray) for item in value):
                    stacked_data = np.stack(value, axis=0)
                    self._add_to_batch_buffer(key, stacked_data)
                else:
                    # Non-array list - store as is
                    self._add_to_batch_buffer(key, value)
            elif isinstance(value, np.ndarray):
                # Handle numpy arrays
                self._add_to_batch_buffer(key, value)
            else:
                # Handle other data types
                self._add_to_batch_buffer(key, value)
    
    def _add_to_batch_buffer(self, key: str, value: Any):
        """Add data to batch buffer"""
        if key not in self.batch_buffer:
            self.batch_buffer[key] = []
        self.batch_buffer[key].append(value)
    
    def _write_batch(self):
        """Write batch buffer to HDF5 file"""
        if not self.batch_buffer:
            return
        
        # Ensure HDF5 file is open
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'w')
        
        for key, batch_data in self.batch_buffer.items():
            if not batch_data:
                continue
            
            # Convert batch to numpy array
            if isinstance(batch_data[0], np.ndarray):
                # Stack arrays along time dimension
                batch_array = np.stack(batch_data, axis=0)
            else:
                # Handle non-array data
                batch_array = np.array(batch_data)
            
            # Write to dataset
            self._write_to_dataset(key, batch_array)
        
        # Clear batch buffer
        self.batch_buffer = {}
    
    def _write_to_dataset(self, key: str, data: np.ndarray):
        """Write data to HDF5 dataset with optimal settings"""
        if data is None or data.dtype == object:
            return
        
        if key not in self.datasets:
            # Create dataset on first write
            self._create_optimized_dataset(key, data)
        
        # Calculate write position
        start_idx = self.frame_counter - self.batch_size
        end_idx = self.frame_counter
        
        # Resize dataset if needed
        dataset = self.datasets[key]
        if end_idx > dataset.shape[0]:
            new_size = max(end_idx, dataset.shape[0] * 2)
            dataset.resize((new_size,) + dataset.shape[1:])
        
        # Write batch data
        dataset[start_idx:end_idx] = data
    
    def _create_optimized_dataset(self, key: str, sample_data: np.ndarray):
        """Create HDF5 dataset with optimal settings for performance"""
        # Calculate optimal chunk shape
        chunk_shape = self._calculate_optimal_chunk_shape(sample_data.shape)
        
        # Create resizable dataset
        maxshape = (None,) + sample_data.shape
        
        # Choose compression settings
        compression_kwargs = {}
        if self.compression == 'lzf':
            compression_kwargs = {'compression': 'lzf'}
        elif self.compression == 'gzip':
            compression_kwargs = {
                'compression': 'gzip',
                'compression_opts': self.compression_opts or 6
            }
        elif self.compression == 'szip':
            compression_kwargs = {
                'compression': 'szip',
                'compression_opts': ('nn', 16)
            }
        
        # Create dataset
        self.datasets[key] = self.h5_file.create_dataset(
            key,
            shape=(self.chunk_size,) + sample_data.shape,
            maxshape=maxshape,
            chunks=chunk_shape,
            dtype=sample_data.dtype,
            **compression_kwargs
        )
        
        if self.verbose:
            print(f"Created optimized dataset '{key}': shape={sample_data.shape}, dtype={sample_data.dtype}, chunks={chunk_shape}")
    
    def _calculate_optimal_chunk_shape(self, data_shape: tuple) -> tuple:
        """Calculate optimal chunk shape for performance"""
        # Start with chunk size for time dimension
        chunk_shape = (self.chunk_size,)
        
        # Add other dimensions
        for dim in data_shape:
            # Use full dimension for small sizes, or reasonable chunk size
            if dim <= 64:
                chunk_shape += (dim,)
            else:
                # Use power of 2 close to 64 for larger dimensions
                chunk_size = min(64, 2 ** int(np.log2(dim)))
                chunk_shape += (chunk_size,)
        
        return chunk_shape
    
    def finish(self, metadata: Optional[Dict] = None):
        """Finish recording and write remaining data to disk"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.verbose:
            spinner = Halo(text='Writing final batch to h5 file...', spinner='dots')
            spinner.start()
        
        # Write any remaining batch data
        if self.batch_buffer:
            self._write_batch()
        
        # Ensure HDF5 file is open
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, 'w')
        
        # Resize datasets to actual size
        for dataset in self.datasets.values():
            dataset.resize((self.frame_counter,) + dataset.shape[1:])
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                try:
                    self.h5_file.attrs[key] = value
                except Exception as e:
                    if self.verbose:
                        print(f"Could not save metadata '{key}': {e}")
        
        # Add writer-specific metadata
        if self.h5_file:
            self.h5_file.attrs['writer_type'] = 'optimized'
            self.h5_file.attrs['chunk_size'] = self.chunk_size
            self.h5_file.attrs['compression'] = self.compression
            self.h5_file.attrs['total_frames'] = self.frame_counter
        
        # Close file
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None
        
        if self.verbose:
            spinner.stop_and_persist(symbol="✅", text="Optimized h5 file written to disk!")
            self._print_performance_stats()
    
    def _print_performance_stats(self):
        """Print performance statistics"""
        if not self.save_times:
            return
        
        avg_save_time = np.mean(self.save_times) * 1000
        max_save_time = np.max(self.save_times) * 1000
        min_save_time = np.min(self.save_times) * 1000
        total_frames = self.frame_counter
        
        print(colored("-" * 50, "blue"))
        print(colored("Optimized DataWriter Performance Stats:", "blue"))
        print(colored(f"Total frames processed: {total_frames}", "blue"))
        print(colored(f"Average save time: {avg_save_time:.2f} ms", "blue"))
        print(colored(f"Min save time: {min_save_time:.2f} ms", "blue"))
        print(colored(f"Max save time: {max_save_time:.2f} ms", "blue"))
        print(colored(f"Estimated FPS: {1000/avg_save_time:.1f}", "blue"))
        print(colored(f"Chunk size: {self.chunk_size}", "blue"))
        print(colored(f"Compression: {self.compression}", "blue"))
        print(colored(f"Datasets created: {len(self.datasets)}", "blue"))
        print(colored("-" * 50, "blue"))
    
    def add_info(self, **data):
        """Add header information (compatible with original DataWriter)"""
        if self.h5_file is None:
            raise RuntimeError('no file')
        for k, v in data.items():
            try:
                self.h5_file.attrs[k] = v
            except Exception as e:
                if self.verbose:
                    print(f"Could not save info '{k}': {e}")
    
    def add_attributes(self, **data):
        """Add attributes (compatible with original DataWriter)"""
        if self.h5_file is None:
            raise RuntimeError('no file')
        for k, v in data.items():
            try:
                self.h5_file.attrs[k] = v
            except Exception as e:
                if self.verbose:
                    print(f"Could not save attribute '{k}': {e}")


# Utility function to create optimized writer
def create_optimized_writer(directory: str, **kwargs) -> OptimizedDataWriter:
    """Create an optimized data writer with the same interface as the original"""
    return OptimizedDataWriter(directory, **kwargs)