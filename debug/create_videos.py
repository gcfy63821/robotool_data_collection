
import os
import cv2
import h5py
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm
import subprocess


def write_video_ffmpeg(images, out_path, fps=25, crf=23, preset="medium"):
    """Write frames directly to H.264-compressed MP4 using ffmpeg."""
    height, width, _ = images[0].shape

    cmd = [
        "ffmpeg",
        "-y",                        # overwrite existing file
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",                   # stdin as input
        "-an",                       # no audio
        "-vcodec", "libx264",
        "-preset", preset,           # speed/size tradeoff: ultrafast, fast, medium, slow
        "-crf", str(crf),            # quality/size tradeoff: 18–28 typical
        out_path,
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for img in images:
        bgr = (img * 255).clip(0, 255).astype(np.uint8)
        process.stdin.write(bgr.tobytes())

    process.stdin.close()
    process.wait()


def create_grid_video(view_to_images, out_path, fps=25, crf=23, preset="medium"):
    """Create a video with all views arranged in a 4x2 grid."""
    if not view_to_images:
        return
    
    # Get dimensions from first image
    first_view = list(view_to_images.keys())[0]
    single_height, single_width, _ = view_to_images[first_view][0].shape
    
    # Create 4x2 grid
    grid_width = single_width * 4
    grid_height = single_height * 2
    
    # Get total frames (use the view with most frames)
    max_frames = max(len(images) for images in view_to_images.values())
    
    # Create grid frames
    grid_frames = []
    for frame_idx in range(max_frames):
        # Create empty grid
        grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.float32)
        
        # Fill grid positions (4x2 layout)
        grid_positions = [
            (0, 0), (0, 1), (0, 2), (0, 3),  # Top row
            (1, 0), (1, 1), (1, 2), (1, 3)   # Bottom row
        ]
        
        for view_idx, (row, col) in enumerate(grid_positions):
            if view_idx in view_to_images and frame_idx < len(view_to_images[view_idx]):
                # Calculate position in grid
                y_start = row * single_height
                y_end = y_start + single_height
                x_start = col * single_width
                x_end = x_start + single_width
                
                # Place image in grid
                grid_frame[y_start:y_end, x_start:x_end] = view_to_images[view_idx][frame_idx]
        
        grid_frames.append(grid_frame)
    
    # Write grid video
    write_video_ffmpeg(grid_frames, out_path, fps, crf, preset)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/robot/drive/robotool/videos_0902/woodenfork_greenbowl_stir_coffee")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default = "/home/robot/drive/robotool/videos_0902_annotated/woodenfork_greenbowl_stir_coffee")
    parser.add_argument("--save_video", action="store_true", default="True")
    parser.add_argument("--num", type=int, default = 0)
    args = parser.parse_args()
    for f in (os.listdir(args.data_root)):
        fold = os.path.join(args.data_root, f)
        if not os.path.isdir(fold):
            print(f"Skipping {f}, not a directory.")
            continue
        print(fold)
        h5_file = os.path.join(fold, "data00000000.h5")
        # if not fold.startswith(str('/home/robot/drive/robotool/videos_0830/133f17ce')):
        #     print("skipping")
        #     continue
        if not os.path.exists(h5_file):
            print(f"Skipping {f}, no data00000000.h5 found.")
            continue
        video_dir = os.path.join(args.output_dir, f, "videos")
        print(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        session_h5 = h5py.File(h5_file, "r")
        total_frames = len(session_h5["imgs"])
        start_frame = args.start_frame
        end_frame = args.end_frame if args.end_frame is not None else total_frames
        view_to_images = defaultdict(list)
        for frame_idx in tqdm(range(start_frame, end_frame)):
            rgbs = session_h5["imgs"][frame_idx]
            if rgbs.ndim == 3:
                rgb = rgbs.astype(np.float32) / 255.0
                view_to_images[0].append(rgb)
            else:
                for view_idx, rgb in enumerate(rgbs):
                    rgb = rgb.astype(np.float32) / 255.0
                    view_to_images[view_idx].append(rgb)
        session_h5.close()
        if args.save_video:
            # Create individual camera videos
            for view, images in tqdm(view_to_images.items()):
                out_path = os.path.join(video_dir, f"cam{view:02d}.mp4")
                write_video_ffmpeg(images, out_path)
            
            # Create grid video with all views
            grid_out_path = os.path.join(video_dir, "all_views_grid.mp4")
            create_grid_video(view_to_images, grid_out_path)