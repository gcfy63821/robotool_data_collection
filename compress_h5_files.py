#!/usr/bin/env python3
"""
Compress H5 files to reduce their size.
This script recompresses datasets in H5 files with different compression settings to reduce file size.
"""

import os
import glob
import argparse
import h5py
import numpy as np
from termcolor import colored
from datetime import datetime
import shutil


def find_h5_files(base_dir):
    """Find all H5 files in the base directory and subdirectories"""
    h5_files = []
    
    # First, check if base_dir itself contains H5 files
    for file in os.listdir(base_dir):
        if file.endswith('.h5'):
            h5_files.append(os.path.join(base_dir, file))
    
    # Then check all subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Skip the base directory itself (already checked above)
        if root == base_dir:
            continue
            
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    
    return h5_files


def find_session_folders(base_dir):
    """Find all session folders that contain H5 files"""
    session_folders = []
    
    # Check if base_dir itself contains H5 files
    if any(file.endswith('.h5') for file in os.listdir(base_dir)):
        session_folders.append(base_dir)
    
    # Check all subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Skip the base directory itself (already checked above)
        if root == base_dir:
            continue
            
        # Check if this directory contains H5 files
        h5_files_in_dir = [f for f in files if f.endswith('.h5')]
        if h5_files_in_dir:
            session_folders.append(root)
    
    return session_folders


def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)


def analyze_h5_file(file_path):
    """Analyze H5 file and return information about datasets"""
    try:
        with h5py.File(file_path, 'r') as f:
            info = {
                'file_path': file_path,
                'original_size_mb': get_file_size_mb(file_path),
                'datasets': {},
                'attributes': dict(f.attrs)
            }
            
            for key in f.keys():
                dataset = f[key]
                info['datasets'][key] = {
                    'shape': dataset.shape,
                    'dtype': str(dataset.dtype),
                    'compression': dataset.compression,
                    'compression_opts': dataset.compression_opts,
                    'chunks': dataset.chunks,
                    'size_mb': dataset.nbytes / (1024 * 1024)
                }
            
            return info
            
    except Exception as e:
        print(colored(f"Error analyzing {file_path}: {e}", "red"))
        return None


def compress_h5_file(file_path, compression='gzip', compression_opts=9, chunks=None, 
                    backup=True, verbose=True):
    """Compress an H5 file with new compression settings"""
    try:
        if verbose:
            print(colored(f"\nCompressing: {file_path}", "cyan"))
        
        # Get original file info
        original_info = analyze_h5_file(file_path)
        if not original_info:
            return False
        
        original_size = original_info['original_size_mb']
        
        # Create backup if requested
        if backup:
            backup_path = file_path + '.backup'
            if not os.path.exists(backup_path):
                shutil.copy2(file_path, backup_path)
                if verbose:
                    print(f"Created backup: {backup_path}")
        
        # Create temporary file for compression
        temp_path = file_path + '.temp'
        
        with h5py.File(file_path, 'r') as src, h5py.File(temp_path, 'w') as dst:
            # Copy attributes
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            
            # Add compression metadata
            dst.attrs['compression_date'] = datetime.now().isoformat()
            dst.attrs['compression_script'] = 'compress_h5_files.py'
            dst.attrs['original_size_mb'] = original_size
            dst.attrs['compression_method'] = compression
            dst.attrs['compression_opts'] = compression_opts
            
            # Copy and recompress datasets
            for key in src.keys():
                dataset = src[key]
                
                # Determine chunks - preserve original chunks if not specified
                if chunks is None:
                    # Use original chunks or auto-determine if none
                    if dataset.chunks is not None:
                        chunk_shape = dataset.chunks
                    else:
                        # Auto-determine chunks based on dataset shape
                        if len(dataset.shape) >= 3:
                            # For image data, use reasonable chunk size
                            chunk_shape = (1,) + dataset.shape[1:]
                        else:
                            chunk_shape = dataset.shape
                else:
                    chunk_shape = chunks
                
                # Create dataset with new compression but preserve exact data
                dst.create_dataset(
                    key,
                    data=dataset[:],  # Copy exact data
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=chunk_shape
                )
                
                if verbose:
                    print(f"  Compressed dataset '{key}': {dataset.shape} (chunks: {chunk_shape})")
        
        # Replace original with compressed version
        os.replace(temp_path, file_path)
        
        # Get new file size
        new_size = get_file_size_mb(file_path)
        compression_ratio = (1 - new_size / original_size) * 100
        
        if verbose:
            print(colored(f"Compression complete:", "green"))
            print(f"  Original size: {original_size:.2f} MB")
            print(f"  New size: {new_size:.2f} MB")
            print(f"  Size reduction: {compression_ratio:.1f}%")
            print(f"  Data shape preserved: ✓")
        
        return True, original_size, new_size, compression_ratio
        
    except Exception as e:
        print(colored(f"Error compressing {file_path}: {e}", "red"))
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False, 0, 0, 0


def restore_from_backup(file_path, verbose=True):
    """Restore H5 file from backup"""
    backup_path = file_path + '.backup'
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, file_path)
        if verbose:
            print(colored(f"Restored {file_path} from backup", "green"))
        return True
    else:
        if verbose:
            print(colored(f"No backup found for {file_path}", "red"))
        return False


def main():
    parser = argparse.ArgumentParser(description="Compress H5 files to reduce their size")
    parser.add_argument('--base_dir', type=str, required=True,
                       help="Base directory containing session folders with H5 files")
    parser.add_argument('--file', type=str,
                       help="Compress specific H5 file (optional)")
    parser.add_argument('--compression', type=str, default='gzip', 
                       choices=['gzip', 'lzf', 'szip'],
                       help="Compression method (default: gzip)")
    parser.add_argument('--compression_opts', type=int, default=1,
                       help="Compression level (1-9 for gzip, default: 1)")
    parser.add_argument('--chunks', type=str,
                       help="Chunk shape as comma-separated values (e.g., '1,8,480,640')")
    parser.add_argument('--no_backup', action='store_true',
                       help="Don't create backup files")
    parser.add_argument('--verbose', action='store_true', default=True,
                       help="Verbose output (default: True)")
    parser.add_argument('--analyze_only', action='store_true',
                       help="Only analyze files without compressing")
    parser.add_argument('--restore', action='store_true',
                       help="Restore files from backup")
    parser.add_argument('--dry_run', action='store_true',
                       help="Show what would be compressed without actually doing it")
    args = parser.parse_args()
    
    print(colored("=" * 60, "cyan"))
    print(colored("H5 FILE COMPRESSION SCRIPT", "cyan"))
    print(colored("=" * 60, "cyan"))
    
    # Parse chunks argument
    chunk_shape = None
    if args.chunks:
        try:
            chunk_shape = tuple(int(x.strip()) for x in args.chunks.split(','))
        except ValueError:
            print(colored("Error: Invalid chunk shape format. Use comma-separated integers.", "red"))
            return
    
    if args.restore:
        # Restore mode
        if args.file:
            restore_from_backup(args.file, args.verbose)
        else:
            h5_files = find_h5_files(args.base_dir)
            for h5_file in h5_files:
                restore_from_backup(h5_file, args.verbose)
        return
    
    if args.file:
        # Single file mode
        if not os.path.exists(args.file):
            print(colored(f"File not found: {args.file}", "red"))
            return
        
        if args.analyze_only:
            info = analyze_h5_file(args.file)
            if info:
                print(colored(f"\nFile Analysis: {args.file}", "cyan"))
                print(f"Size: {info['original_size_mb']:.2f} MB")
                print(f"Datasets: {list(info['datasets'].keys())}")
                for key, dataset_info in info['datasets'].items():
                    print(f"  {key}: {dataset_info['shape']} ({dataset_info['size_mb']:.2f} MB)")
                    print(f"    Compression: {dataset_info['compression']}")
                    print(f"    Chunks: {dataset_info['chunks']}")
        elif args.dry_run:
            print(colored(f"DRY RUN: Would compress {args.file}", "yellow"))
            print(f"  Compression: {args.compression}")
            print(f"  Compression opts: {args.compression_opts}")
            if chunk_shape:
                print(f"  Chunks: {chunk_shape}")
        else:
            success, orig_size, new_size, ratio = compress_h5_file(
                args.file, args.compression, args.compression_opts, 
                chunk_shape, not args.no_backup, args.verbose
            )
            if success:
                print(colored("Compression completed successfully!", "green"))
            else:
                print(colored("Compression failed!", "red"))
    else:
        # Batch mode - process all session folders
        print(f"Searching for session folders with H5 files in: {args.base_dir}")
        
        if not os.path.exists(args.base_dir):
            print(colored(f"Base directory not found: {args.base_dir}", "red"))
            return
        
        # Find all session folders that contain H5 files
        session_folders = find_session_folders(args.base_dir)
        
        if not session_folders:
            print(colored("No session folders with H5 files found", "yellow"))
            return
        
        print(f"\nFound {len(session_folders)} session folders:")
        for folder in session_folders:
            h5_files_in_folder = [f for f in os.listdir(folder) if f.endswith('.h5')]
            print(f"  {folder}: {len(h5_files_in_folder)} H5 files")
        
        if args.analyze_only:
            total_size = 0
            for folder in session_folders:
                h5_files_in_folder = [f for f in os.listdir(folder) if f.endswith('.h5')]
                for h5_file in h5_files_in_folder:
                    file_path = os.path.join(folder, h5_file)
                    info = analyze_h5_file(file_path)
                    if info:
                        total_size += info['original_size_mb']
                        print(colored(f"\nFile: {file_path}", "cyan"))
                        print(f"Size: {info['original_size_mb']:.2f} MB")
                        print(f"Datasets: {list(info['datasets'].keys())}")
            
            print(colored(f"\nTotal size: {total_size:.2f} MB", "cyan"))
            return
        
        if args.dry_run:
            print(colored("DRY RUN: Would compress H5 files in the following folders:", "yellow"))
            for folder in session_folders:
                h5_files_in_folder = [f for f in os.listdir(folder) if f.endswith('.h5')]
                for h5_file in h5_files_in_folder:
                    print(f"  {os.path.join(folder, h5_file)}")
            print(f"\nCompression settings:")
            print(f"  Method: {args.compression}")
            print(f"  Level: {args.compression_opts}")
            if chunk_shape:
                print(f"  Chunks: {chunk_shape}")
            return
        
        # Compress all files in all session folders
        successful_compressions = 0
        failed_compressions = 0
        total_original_size = 0
        total_new_size = 0
        
        file_counter = 0
        total_files = sum(len([f for f in os.listdir(folder) if f.endswith('.h5')]) for folder in session_folders)
        
        for folder in session_folders:
            h5_files_in_folder = [f for f in os.listdir(folder) if f.endswith('.h5')]
            
            for h5_file in h5_files_in_folder:
                file_counter += 1
                file_path = os.path.join(folder, h5_file)
                
                print(colored(f"\n[{file_counter}/{total_files}] Processing: {file_path}", "cyan"))
                
                success, orig_size, new_size, ratio = compress_h5_file(
                    file_path, args.compression, args.compression_opts, 
                    chunk_shape, not args.no_backup, args.verbose
                )
                
                if success:
                    successful_compressions += 1
                    total_original_size += orig_size
                    total_new_size += new_size
                else:
                    failed_compressions += 1
        
        # Summary
        print(colored("\n" + "=" * 60, "cyan"))
        print(colored("COMPRESSION SUMMARY", "cyan"))
        print(colored("=" * 60, "cyan"))
        print(colored(f"Total session folders: {len(session_folders)}", "cyan"))
        print(colored(f"Total H5 files: {total_files}", "cyan"))
        print(colored(f"Successful compressions: {successful_compressions}", "green"))
        print(colored(f"Failed compressions: {failed_compressions}", "red"))
        
        if successful_compressions > 0:
            total_compression_ratio = (1 - total_new_size / total_original_size) * 100
            print(colored(f"Total original size: {total_original_size:.2f} MB", "cyan"))
            print(colored(f"Total new size: {total_new_size:.2f} MB", "cyan"))
            print(colored(f"Total size reduction: {total_compression_ratio:.1f}%", "green"))
        
        if not args.no_backup:
            print(colored("Backup files created (use --restore to restore)", "blue"))


if __name__ == "__main__":
    main()
