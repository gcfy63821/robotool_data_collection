import os
import h5py
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
import re


def get_video_length_from_h5(h5_file_path):
    """
    Extract video length (number of frames) from an H5 file.
    
    Args:
        h5_file_path: Path to the H5 file
        
    Returns:
        int: Number of frames in the video, or 0 if error
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'imgs' in f:
                total_frames = len(f['imgs'])
                return total_frames
            else:
                print(f"Warning: No 'imgs' dataset found in {h5_file_path}")
                return 0
    except Exception as e:
        print(f"Error reading {h5_file_path}: {e}")
        return 0


def parse_folder_name(folder_name):
    """
    Parse folder name to extract prefix and number suffix.
    
    Args:
        folder_name: Name of the folder
        
    Returns:
        tuple: (prefix, number) or (None, None) if no number found
    """
    # Find the last sequence of digits at the end of the folder name
    match = re.search(r'^(.+?)(\d+)$', folder_name)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return prefix, number
    return None, None


def calculate_video_lengths(data_root, fps=60):
    """
    Calculate total video lengths grouped by folder naming pattern.
    
    Args:
        data_root: Root directory containing video folders
        fps: Frames per second (default: 60)
        
    Returns:
        dict: Dictionary with prefix as key and list of (folder, frames, duration) as value
    """
    video_groups = defaultdict(list)
    
    # Scan all folders in data_root
    for folder_name in os.listdir(data_root):
        parent_folder_path = os.path.join(data_root, folder_name)
        
        if not os.path.isdir(parent_folder_path):
            continue
        for sub_folder_name in os.listdir(parent_folder_path):
            folder_path = os.path.join(parent_folder_path, sub_folder_name)
            # Look for H5 file
            h5_file = os.path.join(folder_path, "data00000000.h5")
            print(h5_file)
            if not os.path.exists(h5_file):
                print(f"Skipping {folder_name}, no data00000000.h5 found.")
                continue
            
            # Parse folder name to get prefix and number
            prefix, number = parse_folder_name(sub_folder_name)
            if prefix is None:
                print(f"Skipping {folder_name}, no number suffix found.")
                continue
            
            # Get video length
            frames = get_video_length_from_h5(h5_file)
            if frames > 0:
                duration_seconds = frames / fps
                duration_minutes = duration_seconds / 60
                
                video_groups[prefix].append({
                    'folder': folder_name,
                    'number': number,
                    'frames': frames,
                    'duration_seconds': duration_seconds,
                    'duration_minutes': duration_minutes
                })
    
    return video_groups


def print_summary(video_groups, fps):
    """
    Print a summary of video lengths grouped by prefix.
    
    Args:
        video_groups: Dictionary from calculate_video_lengths
        fps: Frames per second
    """
    print(f"\n{'='*80}")
    print(f"VIDEO LENGTH SUMMARY (assuming {fps} FPS)")
    print(f"{'='*80}")
    
    total_all_frames = 0
    total_all_duration = 0
    
    for prefix in sorted(video_groups.keys()):
        videos = video_groups[prefix]
        
        # Sort by number for consistent ordering
        videos.sort(key=lambda x: x['number'])
        
        # Calculate totals for this prefix
        total_frames = sum(v['frames'] for v in videos)
        total_duration_seconds = sum(v['duration_seconds'] for v in videos)
        total_duration_minutes = total_duration_seconds / 60
        
        print(f"\n📁 PREFIX: '{prefix}'")
        print(f"   Total videos: {len(videos)}")
        print(f"   Total frames: {total_frames:,}")
        print(f"   Total duration: {total_duration_seconds:.1f}s ({total_duration_minutes:.2f}min)")
        print(f"   Average per video: {total_duration_seconds/len(videos):.1f}s ({total_duration_minutes/len(videos):.2f}min)")
        
        print(f"\n   Individual videos:")
        for video in videos:
            print(f"     {video['folder']:30s} | {video['frames']:6,} frames | {video['duration_seconds']:6.1f}s | {video['duration_minutes']:6.2f}min")
        
        total_all_frames += total_frames
        total_all_duration += total_duration_seconds
    
    # Overall summary
    total_all_minutes = total_all_duration / 60
    total_all_hours = total_all_minutes / 60
    
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Total unique prefixes: {len(video_groups)}")
    print(f"Total videos: {sum(len(videos) for videos in video_groups.values())}")
    print(f"Total frames: {total_all_frames:,}")
    print(f"Total duration: {total_all_duration:.1f}s ({total_all_minutes:.2f}min, {total_all_hours:.2f}hrs)")
    print(f"{'='*80}")


def main():
    parser = ArgumentParser(description="Calculate video lengths grouped by folder naming patterns")
    parser.add_argument("--data_root", type=str, 
                       default="/home/robot/drive/robotool/videos_0903",
                       help="Root directory containing video folders")
    parser.add_argument("--fps", type=int, default=25,
                       help="Frames per second (default: 60)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root directory '{args.data_root}' does not exist.")
        return
    
    print(f"Scanning directory: {args.data_root}")
    print(f"Assuming FPS: {args.fps}")
    
    # Calculate video lengths
    video_groups = calculate_video_lengths(args.data_root, args.fps)
    
    if not video_groups:
        print("No video folders found.")
        return
    
    # Print summary
    print_summary(video_groups, args.fps)


if __name__ == "__main__":
    main()
