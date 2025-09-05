#!/usr/bin/env python3
"""
Convert .npy files to H5 format for all session directories.
This script processes all session directories that contain .npy files and converts them to H5 format.
"""

import os
import glob
import json
import argparse
import h5py
import numpy as np
from termcolor import colored
from datetime import datetime
import shutil


def find_session_directories(base_dir):
    """Find all session directories that contain .npy files"""
    session_dirs = []
    
    # Look for directories with imgs/ and depths/ subdirectories containing .npy files
    for root, dirs, files in os.walk(base_dir):
        if "imgs" in dirs and "depths" in dirs:
            img_files = glob.glob(os.path.join(root, "imgs", "frame_*.npy"))
            depth_files = glob.glob(os.path.join(root, "depths", "frame_*.npy"))
            
            if img_files or depth_files:
                session_dirs.append(root)
                print(f"Found session directory: {root}")
                print(f"  Image files: {len(img_files)}")
                print(f"  Depth files: {len(depth_files)}")
    
    return session_dirs


def load_metadata(session_dir):
    """Load metadata from session directory"""
    metadata_file = os.path.join(session_dir, "metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(colored(f"Error loading metadata from {metadata_file}: {e}", "red"))
    
    # Return default metadata if file doesn't exist
    return {
        "session_name": os.path.basename(session_dir),
        "episode": "unknown",
        "total_frames": 0,
        "target_fps": 25,
        "writer_type": "fast_background"
    }


def convert_session_to_h5(session_dir, cleanup=False, verbose=True):
    """Convert a single session directory from .npy to H5 format"""
    try:
        if verbose:
            print(colored(f"\nConverting session: {session_dir}", "cyan"))
        
        # Load metadata
        metadata = load_metadata(session_dir)
        
        # Get all frame files
        img_files = sorted(glob.glob(os.path.join(session_dir, "imgs", "frame_*.npy")))
        depth_files = sorted(glob.glob(os.path.join(session_dir, "depths", "frame_*.npy")))
        
        if not img_files and not depth_files:
            print(colored(f"No .npy files found in {session_dir}", "yellow"))
            return False
        
        if verbose:
            print(f"Found {len(img_files)} image files and {len(depth_files)} depth files")
        
        # Create H5 file
        h5_file = os.path.join(session_dir, "data00000000.h5")
        
        with h5py.File(h5_file, 'w') as f:
            # Load and stack all images
            if img_files:
                if verbose:
                    print("Loading images...")
                imgs_list = []
                for img_file in img_files:
                    img_data = np.load(img_file)
                    imgs_list.append(img_data)
                
                if imgs_list:
                    imgs_array = np.stack(imgs_list, axis=0)
                    f.create_dataset('imgs', data=imgs_array, compression='gzip', compression_opts=1)
                    if verbose:
                        print(f"Saved {len(imgs_list)} image frames with shape {imgs_array.shape}")
            
            # Load and stack all depths
            if depth_files:
                if verbose:
                    print("Loading depths...")
                depths_list = []
                for depth_file in depth_files:
                    depth_data = np.load(depth_file)
                    depths_list.append(depth_data)
                
                if depths_list:
                    depths_array = np.stack(depths_list, axis=0)
                    f.create_dataset('depths', data=depths_array, compression='gzip', compression_opts=1)
                    if verbose:
                        print(f"Saved {len(depths_list)} depth frames with shape {depths_array.shape}")
            
            # Add metadata
            for key, value in metadata.items():
                try:
                    f.attrs[key] = value
                except Exception as e:
                    if verbose:
                        print(f"Could not save metadata '{key}': {e}")
            
            # Add conversion metadata
            f.attrs['conversion_date'] = datetime.now().isoformat()
            f.attrs['conversion_script'] = 'convert_npy_to_h5.py'
            f.attrs['original_format'] = 'npy_files'
            f.attrs['total_frames'] = len(imgs_list) if img_files else 0
        
        if verbose:
            print(colored(f"Successfully converted to H5: {h5_file}", "green"))
        
        # Clean up .npy files if requested
        if cleanup:
            cleanup_npy_files(session_dir, verbose)
        
        return True
        
    except Exception as e:
        print(colored(f"Error converting session {session_dir}: {e}", "red"))
        return False


def cleanup_npy_files(session_dir, verbose=True):
    """Clean up .npy files and directories after conversion"""
    try:
        # Remove .npy files
        img_files = glob.glob(os.path.join(session_dir, "imgs", "*.npy"))
        depth_files = glob.glob(os.path.join(session_dir, "depths", "*.npy"))
        
        for img_file in img_files:
            os.remove(img_file)
        for depth_file in depth_files:
            os.remove(depth_file)
        
        # Remove empty directories
        imgs_dir = os.path.join(session_dir, "imgs")
        depths_dir = os.path.join(session_dir, "depths")
        
        if os.path.exists(imgs_dir) and not os.listdir(imgs_dir):
            os.rmdir(imgs_dir)
        if os.path.exists(depths_dir) and not os.listdir(depths_dir):
            os.rmdir(depths_dir)
        
        # Remove metadata.json
        metadata_file = os.path.join(session_dir, "metadata.json")
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        
        if verbose:
            print(colored(f"Cleaned up {len(img_files)} image files and {len(depth_files)} depth files", "green"))
            
    except Exception as e:
        print(colored(f"Error cleaning up .npy files: {e}", "red"))


def main():
    parser = argparse.ArgumentParser(description="Convert .npy files to H5 format for all session directories")
    parser.add_argument('--base_dir', type=str, default="/home/robot/drive/robotool/videos_0903",
                       help="Base directory containing session directories (default: /home/robot/drive/robotool/videos_0903)")
    parser.add_argument('--session_dir', type=str, 
                       help="Convert specific session directory (optional)")
    parser.add_argument('--cleanup', action='store_true', 
                       help="Clean up .npy files after conversion")
    parser.add_argument('--verbose', action='store_true', default=True,
                       help="Verbose output (default: True)")
    parser.add_argument('--dry_run', action='store_true',
                       help="Show what would be converted without actually converting")
    args = parser.parse_args()
    
    print(colored("=" * 60, "cyan"))
    print(colored("NPY TO H5 CONVERSION SCRIPT", "cyan"))
    print(colored("=" * 60, "cyan"))
    
    if args.session_dir:
        # Convert specific session directory
        if not os.path.exists(args.session_dir):
            print(colored(f"Session directory not found: {args.session_dir}", "red"))
            return
        
        if args.dry_run:
            print(colored(f"DRY RUN: Would convert {args.session_dir}", "yellow"))
            return
        
        success = convert_session_to_h5(args.session_dir, args.cleanup, args.verbose)
        if success:
            print(colored("Conversion completed successfully!", "green"))
        else:
            print(colored("Conversion failed!", "red"))
    else:
        # Find and convert all session directories
        print(f"Searching for session directories in: {args.base_dir}")
        
        if not os.path.exists(args.base_dir):
            print(colored(f"Base directory not found: {args.base_dir}", "red"))
            return
        
        session_dirs = find_session_directories(args.base_dir)
        
        if not session_dirs:
            print(colored("No session directories with .npy files found", "yellow"))
            return
        
        print(f"\nFound {len(session_dirs)} session directories to convert")
        
        if args.dry_run:
            print(colored("DRY RUN: Would convert the following directories:", "yellow"))
            for session_dir in session_dirs:
                print(f"  {session_dir}")
            return
        
        # Convert all sessions
        successful_conversions = 0
        failed_conversions = 0
        
        for i, session_dir in enumerate(session_dirs, 1):
            print(colored(f"\n[{i}/{len(session_dirs)}] Processing: {session_dir}", "cyan"))
            
            success = convert_session_to_h5(session_dir, args.cleanup, args.verbose)
            if success:
                successful_conversions += 1
            else:
                failed_conversions += 1
        
        # Summary
        print(colored("\n" + "=" * 60, "cyan"))
        print(colored("CONVERSION SUMMARY", "cyan"))
        print(colored("=" * 60, "cyan"))
        print(colored(f"Total sessions: {len(session_dirs)}", "cyan"))
        print(colored(f"Successful conversions: {successful_conversions}", "green"))
        print(colored(f"Failed conversions: {failed_conversions}", "red"))
        
        if args.cleanup:
            print(colored("Intermediate .npy files have been cleaned up", "blue"))
        else:
            print(colored("Intermediate .npy files preserved (use --cleanup to remove)", "blue"))


if __name__ == "__main__":
    main()
