import os
import h5py
import numpy as np
from argparse import ArgumentParser


def verify_h5_structure(h5_file_path):
    """Verify the structure and content of an H5 file."""
    print(f"File: {h5_file_path}")
    
    if not os.path.exists(h5_file_path):
        print(f"ERROR: File does not exist!")
        return False
    
    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            # Store dataset information for frame count verification
            datasets_info = {}
            frame_counts = {}
            
            # Collect basic info first
            for key in h5_file.keys():
                dataset = h5_file[key]
                datasets_info[key] = dataset
                
                if isinstance(dataset, h5py.Dataset) and dataset.ndim >= 1:
                    frame_count = dataset.shape[0]
                    frame_counts[key] = frame_count
            
            # Check frame count consistency
            if frame_counts:
                unique_frame_counts = set(frame_counts.values())
                if len(unique_frame_counts) > 1:
                    print(f"!!!WARNING: Inconsistent frame counts: {sorted(unique_frame_counts)}")
                else:
                    print(f"✓ Frame count: {list(unique_frame_counts)[0]}")
            
            # Show simplified dataset info
            print(f"\nDatasets:")
            for key, dataset in datasets_info.items():
                if isinstance(dataset, h5py.Dataset):
                    frame_count = frame_counts.get(key, "N/A")
                    print(f"  {key}: shape={dataset.shape}, frames={frame_count}")
                    
                    # Get min/max values
                    try:
                        if dataset.size > 0:
                            min_val = dataset.min()
                            max_val = dataset.max()
                            print(f"    min: {min_val:.6f}, max: {max_val:.6f}")
                        else:
                            print(f"    empty dataset")
                    except Exception as e:
                        print(f"    error reading values: {e}")
                else:
                    print(f"  {key}: {type(dataset).__name__}")
            
            return True
                
    except Exception as e:
        print(f"ERROR: Failed to read H5 file: {str(e)}")
        return False


def main():
    parser = ArgumentParser(description="Verify H5 file content and structure")
    parser.add_argument("--data_root", type=str, 
                       default="/home/robot/drive/robotool/videos_0903",
                       help="Root directory containing H5 files")
    parser.add_argument("--h5_file", type=str, default=None,
                       help="Specific H5 file to verify (if not specified, will check all in data_root)")
    
    args = parser.parse_args()
    
    if args.h5_file:
        # Verify specific file
        verify_h5_structure(args.h5_file)
    else:
        # Verify all H5 files in data_root
        print(f"Scanning directory: {args.data_root}")
        
        if not os.path.exists(args.data_root):
            print(f"ERROR: Directory {args.data_root} does not exist!")
            return
        
        h5_files_found = []
        for root, dirs, files in os.walk(args.data_root):
            for file in files:
                if file.endswith('.h5'):
                    h5_path = os.path.join(root, file)
                    h5_files_found.append(h5_path)
        
        if not h5_files_found:
            print(f"No H5 files found in {args.data_root}")
            return
        
        print(f"Found {len(h5_files_found)} H5 files:")
        for h5_file in h5_files_found:
            print(f"  {h5_file}")
        
        print(f"\n{'='*80}")
        print(f"Starting verification of {len(h5_files_found)} H5 files...")
        print(f"{'='*80}")
        
        for i, h5_file in enumerate(h5_files_found, 1):
            print(f"\n[{i}/{len(h5_files_found)}] Processing: {h5_file}")
            success = verify_h5_structure(h5_file)
            if not success:
                print(f"!!!Verification failed for {h5_file}")
        
        print(f"\n{'='*80}")
        print(f"Verification completed for {len(h5_files_found)} H5 files")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
