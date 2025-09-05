import json
import struct
import time

import cv2
import numpy as np
import redis


class CameraRedisPubInterface:
    """
    This is the Python Interface for writing img info / img data buffer to redis server.
    """

    def __init__(self, redis_host="172.16.0.1", redis_port=6379, camera_id=0):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.info_redis = redis.StrictRedis(
            host=self.redis_host,
            port=self.redis_port,
            charset="utf-8",
            decode_responses=True,
        )

        self.img_redis = redis.StrictRedis(
            host=self.redis_host,
            port=self.redis_port,
            charset="utf-8",
            decode_responses=False,
        )

        self.camera_name = f"camera_{camera_id}"
        self.camera_id = camera_id

        for key in self.info_redis.scan_iter(f"{self.camera_name}*"):
            self.info_redis.delete(key)

        self.count = 0
        self.save_img = False

    def set_img_info(self, img_info):
        """
        Args:
           img_info (dict): dictionary of image information
        """
        json_img_color_str = json.dumps(img_info)
        self.info_redis.set(f"{self.camera_name}::last_img_info", json_img_color_str)

    def set_img_buffer(self, imgs):
        if "color" in imgs:
            img_color = imgs["color"]
            h, w, c = imgs["color"].shape
            shape = struct.pack(">III", h, w, c)
            encoded_color = shape + imgs["color"].tobytes()
            self.img_redis.set(f"{self.camera_name}::last_img_color", encoded_color)
        if "depth" in imgs:
            h, w = imgs["depth"].shape
            shape = struct.pack(">II", h, w)
            encoded_depth = shape + imgs["depth"].tobytes()
            self.img_redis.set(f"{self.camera_name}::last_img_depth", encoded_depth)
        if "unaligned_depth" in imgs:
            h, w = imgs["unaligned_depth"].shape
            shape = struct.pack(">II", h, w)
            encoded_depth = shape + imgs["unaligned_depth"].tobytes()
            self.img_redis.set(
                f"{self.camera_name}::last_img_unaligned_depth", encoded_depth
            )
        if "points" in imgs:
            n, c = imgs["points"].shape
            shape = struct.pack(">II", n, c)
            encoded_points = shape + imgs["points"].tobytes()
            self.img_redis.set(f"{self.camera_name}::last_img_points", encoded_points)

    def get_save_img_info(self):
        self.save_img = self.info_redis.get(f"{self.camera_name}::save")
        return bool(self.save_img)

    @property
    def finished(self):
        return self.info_redis.get(f"{self.camera_name}::finish")


class CameraRedisSubInterface:
    """
    This is the Python Interface for getting image name from redis server. You need to make sure that camera processes are running and images are published to redis server.
    """

    def __init__(
        self,
        redis_host="172.16.0.1",
        redis_port=6379,
        camera_id=0,
        use_color=True,
        use_depth=False,
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.info_redis = redis.StrictRedis(
            host=self.redis_host,
            port=self.redis_port,
            charset="utf-8",
            decode_responses=True,
        )

        self.img_redis = redis.StrictRedis(
            host=self.redis_host,
            port=self.redis_port,
            charset="utf-8",
            decode_responses=False,
        )

        self.camera_name = f"camera_{camera_id}"
        self.camera_id = camera_id
        for key in self.info_redis.scan_iter(f"{self.camera_name}*"):
            self.info_redis.delete(key)

        self.use_color = use_color
        self.use_depth = use_depth

        self.camera_type = None

    def start(self, timeout=5):
        start_time = time.time()
        end_time = start_time
        while end_time - start_time < timeout:
            json_str = self.info_redis.get(f"{self.camera_name}::last_img_info")
            if json_str is not None:
                for _ in range(5):
                    self.save_img(flag=True)
                    time.sleep(0.02)

                img_info = self.get_img_info()
                self.camera_type = img_info["camera_type"]
                return True
            end_time = time.time()

        raise ValueError(f"No information received from " + self.camera_name)

    def stop(self):
        for _ in range(5):
            self.save_img(flag=False)
            time.sleep(0.02)

    def save_img(self, flag=False):
        if flag:
            self.info_redis.set(f"{self.camera_name}::save", 1)
        else:
            self.info_redis.delete(f"{self.camera_name}::save")

    def get_img_info(self):
        img_info = self.info_redis.get(f"{self.camera_name}::last_img_info")
        if img_info is not None:
            img_info = json.loads(img_info)
        return img_info

    def get_img(self):

        img_color = None
        img_depth = None
        img_points = None
        if self.use_color:
            color_buffer = self.img_redis.get(f"{self.camera_name}::last_img_color")
            h, w, c = struct.unpack(">III", color_buffer[:12])
            img_color = np.frombuffer(color_buffer[12:], dtype=np.uint8).reshape(
                h, w, c
            )
        if self.use_depth:
            depth_buffer = self.img_redis.get(f"{self.camera_name}::last_img_depth")
            h, w = struct.unpack(">II", depth_buffer[:8])
            img_depth = np.frombuffer(depth_buffer[8:], dtype=np.uint16).reshape(h, w)

            points_buffer = self.img_redis.get(f"{self.camera_name}::last_img_points")
            n, c = struct.unpack(">II", points_buffer[:8])
            img_points = np.frombuffer(points_buffer[8:], dtype=np.float32).reshape(
                n, c
            )

        return {
            "timestamp": time.time_ns(),
            "color": cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB),
            "depth": img_depth,
            "points": img_points
        }

    def finish(self):
        for _ in range(10):
            self.save_img(flag=False)
            time.sleep(0.02)
        self.info_redis.set(f"{self.camera_name}::finish", 1)

    def close(self):
        self.finish()
