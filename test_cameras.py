from typing import List, Optional, Tuple

import time
import numpy as np
import trimesh
from halo import Halo
import pyrealsense2 as rs
from termcolor import colored
import cv2
import os
import shutil
import uuid
import argparse

import sys
sys.path.insert(0, "./")
from src.robotool.observables.camera.multi_realsense import MultiRealsense
from src.robotool.observables.observable import PointCloudAndRobotObservable
from src.robotool.io.data_writer import DataWriter

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="name of the sequence")
    args = parser.parse_args()

    camera_serial = ["244222072252", 
                     "250122071059",
                     "125322060645",
                     "246422071990",
                     # "123122060454",
                     "246422071818",
                     "246422070730",
                     "250222072777",
                     "204222063088"]
    
    num_cameras = len(camera_serial)
    _img_scale = 1
    _desired_fps = 25

    session_id = str(uuid.uuid4())[:8]
    

    observable = PointCloudAndRobotObservable(
        camera_ids=camera_serial,
        camera_intrinsics=None,
        camera_transformations=None,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
    )

    observable.start()

    idx = 0

    while True:
        
        session_dir = os.path.join("/home/robot/drive/robotool/videos_0817", f"{session_id}_{args.name}_{idx}")
        os.makedirs(session_dir)
        writer = DataWriter(session_dir)


        observable.reset()

        writer.start(0)
        
        prev_time = time.time()
        aborted = False
        # Initialize videos for sam segmentation
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fps = 30
        # video_session_dir = os.path.join("/home/robot/drive/robotool/sam_videos_0323", f"{session_id}_{args.name}")
        # os.makedirs(os.path.join("/home/robot/drive/robotool/sam_videos_0323", f"{session_id}_{args.name}"), exist_ok=True)
        # videos = [cv2.VideoWriter(f"{video_session_dir}/view_{i}.mp4", fourcc, fps, (840, 680)) for i in range(len(camera_serial))]
        for i in range(30000):
            obs = observable.get_obs(get_points=False, depth=True, infrared=False)
            
            # Skip the first 50 frames for realsense to warm up
            if i > 50:
                # writer.save_observation(obs)
            
            # visualization
            # tile_image = np.zeros((num_cameras, 480 // int(1 / _img_scale), 640 // int(1 / _img_scale), 3), dtype=np.uint8)
            # for cam_id in range(num_cameras):
            #     rgb = obs["imgs"][cam_id]
            #     tile_image[cam_id] = rgb
            #     frame = np.ones((680, 840, 3)) * 255
            #     frame[100:-100, 100:-100, :] = rgb.copy()
            #     videos[cam_id].write(frame.astype(np.uint8))
            # flat_image = np.vstack((
            #     np.hstack((tile_image[0], tile_image[1], tile_image[2], tile_image[3])),
            #     np.hstack((tile_image[4], tile_image[5], tile_image[6], tile_image[7])),
            # ))
            # # draw_label(flat_image, f"Session ID: {session_id}, Recording in progress", (10, 20), (0, 255, 0))
            # cv2.imshow("rgb", flat_image)
                images = obs["imgs"][:8]  # first 8 images
                rows = 2
                cols = 4

                # Resize all images if needed to same shape (e.g., 100x100 for display)
                resized_imgs = [cv2.resize(img, (200, 200)) for img in images]

                # Stack into a grid
                row_imgs = [np.hstack(resized_imgs[i*cols:(i+1)*cols]) for i in range(rows)]
                grid_img = np.vstack(row_imgs)

                cv2.imshow("Image Grid", grid_img)
                # cv2.imshow("rgb", obs["imgs"][0])
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == ord("s"):
                    print("Stopping, saving data...")
                    print("Session ID: ", session_id)
                    idx += 1
                    # for video in videos:
                    #     video.release()
                    break

                if key == ord("a"):
                    print(colored("\nAborting, NOT saving data...", "red"))
                    print("Session ID: ", session_id)
                    aborted = True
                    break

            # compute the capture rate
            current_time = time.time()
            dt = current_time - prev_time
            wait_time = (1.0 / _desired_fps) - dt
            if wait_time > 0:
                time.sleep(wait_time)
            current_time = time.time()
            print(f"\rCurrent capture rate {round(1 / (current_time - prev_time), 3)} fps", end=" ")
            prev_time = current_time

        if not aborted:
            writer.finish(
                {
                    "session_name": f"{session_id}"
                }
            )
        else:
            shutil.rmtree(session_dir)

        print("Continue???")
        do_next = False

        while True:

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q") or key == ord("s"):
                do_next = True
                print("Continue")
                break

            if key == ord("a"):
                do_next = False
                print("Stop")
                break

        if do_next == False:
            break



    
    observable.stop()