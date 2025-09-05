import argparse
import cv2
import numpy as np
import pupil_apriltags as apriltag
import time
import os
from termcolor import colored
import sys

# Add the parent directory to path to import from calibrate.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibrate import (
    AprilTag, 
    get_mat, 
    to_homogeneous, 
    mat2quat, 
    convert_quat, 
    quat2mat, 
    averageQuaternions, 
    comp_avg_pose,
    DIST_FROM_ORIGIN
)

# Same relative poses as in calibrate.py
REL_POSE_FROM_APRIL_TAG_COORDINATE_ORIGIN = {
    0: get_mat([-DIST_FROM_ORIGIN, -DIST_FROM_ORIGIN, 0], [0, 0, 0]),
    1: get_mat([DIST_FROM_ORIGIN, -DIST_FROM_ORIGIN, 0], [0, 0, 0]),
    2: get_mat([-DIST_FROM_ORIGIN, DIST_FROM_ORIGIN, 0], [0, 0, 0]),
    3: get_mat([DIST_FROM_ORIGIN, DIST_FROM_ORIGIN, 0], [0, 0, 0]),
}

class VideoAprilTagDetector:
    def __init__(self, video_path, tag_size=0.05, camera_intrinsics=None, output_dir="debug_output"):
        self.video_path = video_path
        self.tag_size = tag_size
        self.camera_intrinsics = camera_intrinsics
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize AprilTag detector
        self.april_tag = AprilTag(tag_size)
        
        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(colored(f"Video loaded: {video_path}", "green"))
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")
        
        # If no intrinsics provided, use default (you may need to adjust these)
        if self.camera_intrinsics is None:
            # Default intrinsics - you should replace these with actual camera intrinsics
            fx = fy = 600.0  # Approximate focal length
            ppx = self.width / 2.0
            ppy = self.height / 2.0
            self.camera_intrinsics = [fx, fy, ppx, ppy]
            print(colored(f"Using default camera intrinsics: {self.camera_intrinsics}", "yellow"))
        
        # Results storage
        self.detection_results = []
        self.camera_transforms = []
        
    def detect_tags_in_frame(self, frame, frame_idx):
        """Detect AprilTags in a single frame"""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Detect tags
        detections = self.april_tag.detect(gray, self.camera_intrinsics)
        
        # Filter for our specific tag IDs (0, 1, 2, 3)
        valid_detections = [det for det in detections if det.tag_id in [0, 1, 2, 3]]
        
        # Calculate camera transformation if we have enough tags
        camera_transform = None
        if len(valid_detections) >= 2:  # Need at least 2 tags for transformation
            try:
                camera_transform = self.calculate_camera_transform(valid_detections)
            except Exception as e:
                print(colored(f"Error calculating transform for frame {frame_idx}: {e}", "red"))
        
        return valid_detections, camera_transform
    
    def calculate_camera_transform(self, detections):
        """Calculate camera transformation relative to the board"""
        cam_to_bases = []
        
        for detection in detections:
            if detection.tag_id in REL_POSE_FROM_APRIL_TAG_COORDINATE_ORIGIN:
                rel_pose = REL_POSE_FROM_APRIL_TAG_COORDINATE_ORIGIN[detection.tag_id]
                # Calculate transform from camera to base
                transform = to_homogeneous(detection.pose_t, detection.pose_R) @ np.linalg.inv(rel_pose)
                cam_to_bases.append(np.linalg.inv(transform))
        
        if len(cam_to_bases) == 0:
            return None
        
        # Average the transformations
        return comp_avg_pose(cam_to_bases)
    
    def draw_detections(self, frame, detections, camera_transform=None, frame_idx=0):
        """Draw AprilTag detections and info on frame"""
        result_frame = frame.copy()
        
        # Draw each detection
        for i, detection in enumerate(detections):
            corners = detection.corners.astype(int)
            
            # Draw bounding box
            cv2.polylines(result_frame, [corners], True, (0, 255, 0), 2)
            
            # Draw tag ID
            center = corners.mean(axis=0).astype(int)
            cv2.putText(result_frame, f"ID:{detection.tag_id}", 
                       (center[0] - 20, center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(result_frame, f"h:{detection.hamming}", 
                       (center[0] - 20, center[1] + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Draw frame info
        cv2.putText(result_frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_frame, f"Tags: {len(detections)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw camera transform info
        if camera_transform is not None:
            # Extract position and rotation
            position = camera_transform[:3, 3]
            rotation = camera_transform[:3, :3]
            
            # Convert rotation to euler angles for display
            euler = self.rotation_matrix_to_euler_angles(rotation)
            
            cv2.putText(result_frame, f"Pos: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(result_frame, f"Rot: [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(result_frame, "No valid transform", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return result_frame
    
    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles (ZYX order)"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def process_video(self, max_frames=None, save_frames=True, show_preview=True):
        """Process the entire video"""
        print(colored("Starting video processing...", "cyan"))
        
        frame_idx = 0
        valid_transforms = []
        
        # Video writer for output
        if save_frames:
            output_video_path = os.path.join(self.output_dir, "detected_tags_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if max_frames and frame_idx >= max_frames:
                    break
                
                # Detect tags in this frame
                detections, camera_transform = self.detect_tags_in_frame(frame, frame_idx)
                
                # Store results
                self.detection_results.append({
                    'frame_idx': frame_idx,
                    'detections': detections,
                    'camera_transform': camera_transform
                })
                
                if camera_transform is not None:
                    valid_transforms.append(camera_transform)
                
                # Draw detections
                result_frame = self.draw_detections(frame, detections, camera_transform, frame_idx)
                
                # Save frame
                if save_frames:
                    out.write(result_frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('AprilTag Detection', result_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print(colored("User requested quit", "yellow"))
                        break
                    elif key == ord(' '):  # Space to pause
                        cv2.waitKey(0)
                
                # Progress update
                if frame_idx % 30 == 0:  # Every 30 frames
                    print(f"Processed {frame_idx}/{self.total_frames} frames, "
                          f"Valid transforms: {len(valid_transforms)}")
                
                frame_idx += 1
        
        finally:
            if save_frames:
                out.release()
            cv2.destroyAllWindows()
        
        # Calculate average camera transformation
        if valid_transforms:
            avg_transform = comp_avg_pose(valid_transforms)
            print(colored(f"\nAverage camera transformation calculated from {len(valid_transforms)} valid frames:", "green"))
            print(avg_transform)
            
            # Save results
            self.save_results(avg_transform, valid_transforms)
        else:
            print(colored("No valid camera transformations found!", "red"))
        
        print(colored(f"Processing complete. Output saved to: {self.output_dir}", "green"))
        return valid_transforms
    
    def save_results(self, avg_transform, all_transforms):
        """Save detection results to files"""
        # Save average transformation
        np.savetxt(os.path.join(self.output_dir, "average_camera_transform.txt"), 
                  avg_transform, fmt='%.6f')
        
        # Save all transformations
        np.save(os.path.join(self.output_dir, "all_camera_transforms.npy"), 
               np.array(all_transforms))
        
        # Save detection summary
        summary_file = os.path.join(self.output_dir, "detection_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"AprilTag Detection Summary\n")
            f.write(f"========================\n")
            f.write(f"Video: {self.video_path}\n")
            f.write(f"Total frames processed: {len(self.detection_results)}\n")
            f.write(f"Frames with valid transforms: {len(all_transforms)}\n")
            f.write(f"Tag size: {self.tag_size}\n")
            f.write(f"Camera intrinsics: {self.camera_intrinsics}\n\n")
            
            f.write(f"Average Camera Transformation:\n")
            f.write(f"{avg_transform}\n\n")
            
            f.write(f"Position: [{avg_transform[0,3]:.6f}, {avg_transform[1,3]:.6f}, {avg_transform[2,3]:.6f}]\n")
            f.write(f"Rotation matrix:\n{avg_transform[:3, :3]}\n")
        
        print(f"Results saved to {self.output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Detect AprilTags in video and calculate camera transformation')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input MP4 video file')
    parser.add_argument('--tag_size', type=float, default=0.05, help='AprilTag size in meters (default: 0.05)')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process')
    parser.add_argument('--output_dir', type=str, default='debug_output', help='Output directory for results')
    parser.add_argument('--no_preview', action='store_true', help='Disable preview window')
    parser.add_argument('--no_save', action='store_true', help='Disable saving output video')
    
    # Camera intrinsics (you may need to adjust these for your camera)
    parser.add_argument('--fx', type=float, default=600.0, help='Camera focal length x')
    parser.add_argument('--fy', type=float, default=600.0, help='Camera focal length y')
    parser.add_argument('--ppx', type=float, default=None, help='Principal point x (default: width/2)')
    parser.add_argument('--ppy', type=float, default=None, help='Principal point y (default: height/2)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(colored(f"Error: Video file not found: {args.video_path}", "red"))
        return
    
    # Set up camera intrinsics
    camera_intrinsics = None
    if args.fx and args.fy:
        # Get video dimensions for principal point
        cap = cv2.VideoCapture(args.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        ppx = args.ppx if args.ppx is not None else width / 2.0
        ppy = args.ppy if args.ppy is not None else height / 2.0
        
        camera_intrinsics = [args.fx, args.fy, ppx, ppy]
        print(f"Using camera intrinsics: {camera_intrinsics}")
    
    # Create detector and process video
    detector = VideoAprilTagDetector(
        video_path=args.video_path,
        tag_size=args.tag_size,
        camera_intrinsics=camera_intrinsics,
        output_dir=args.output_dir
    )
    
    try:
        detector.process_video(
            max_frames=args.max_frames,
            save_frames=not args.no_save,
            show_preview=not args.no_preview
        )
    except KeyboardInterrupt:
        print(colored("\nProcessing interrupted by user", "yellow"))
    except Exception as e:
        print(colored(f"Error during processing: {e}", "red"))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
