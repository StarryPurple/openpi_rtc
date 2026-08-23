import time

import cv2
import threading

import numpy as np

from examples.xtrainer_real.cameras.realsense_camera import RealSenseCamera


image_left, image_right, image_top, thread_run = (
    np.zeros((480, 640, 3), np.uint8),
    np.zeros((480, 640, 3), np.uint8),
    np.zeros((480, 640, 3), np.uint8),
    1,
)
depth_left_raw, depth_right_raw, depth_top_raw = (
    np.zeros((480, 640), np.uint16),
    np.zeros((480, 640), np.uint16),
    np.zeros((480, 640), np.uint16),
)


def run_thread_cam(rs_cam, which_cam):
    global image_left, image_right, image_top, thread_run
    global depth_left_raw, depth_right_raw, depth_top_raw

    if which_cam == 0:
        while thread_run:
            # print("camera debug print")
            image_left, depth = rs_cam.read()
            image_left = image_left[:, :, ::-1]
            depth_left_raw = np.squeeze(depth).copy()
    elif which_cam == 1:
        while thread_run:
            image_right, depth = rs_cam.read()
            image_right = image_right[:, :, ::-1]
            depth_right_raw = np.squeeze(depth).copy()
    elif which_cam == 2:
        while thread_run:
            image_top_src, depth = rs_cam.read()
            image_top_src = image_top_src[150:420, 220:480, ::-1]
            image_top = cv2.resize(image_top_src, (640, 480))
            depth_top_src = np.squeeze(depth)[150:420, 220:480]
            depth_top_raw = cv2.resize(depth_top_src, (640, 480), interpolation=cv2.INTER_NEAREST)

    else:
        print("Camera index error! ")


class ImageRecorder:
    def __init__(self, bool_flip=False):
        self.camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
        # top = 218622271430
        # left = 218622270365
        # right = 218622276272
        rs_left = RealSenseCamera(flip=False, device_id="218622270365")  # left
        rs_right = RealSenseCamera(flip=True, device_id="218622276272")  # right
        rs_top = RealSenseCamera(flip=True, device_id="218622271430")  # top
        self.thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs_left, 0))
        self.thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs_right, 1))
        self.thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs_top, 2))
        self.thread_cam_left.start()
        self.thread_cam_right.start()
        self.thread_cam_top.start()

    def get_images(self):
        return {
            "cam_high": image_top.copy().astype(np.uint8),
            "cam_low": image_top.copy().astype(np.uint8),
            "cam_left_wrist": image_left.copy().astype(np.uint8),
            "cam_right_wrist": image_right.copy().astype(np.uint8),
        }

    def get_rgbd_images(self):
        return {
            "images": self.get_images(),
            "depth_raw": {
                "cam_high": depth_top_raw.copy().astype(np.uint16),
                "cam_left_wrist": depth_left_raw.copy().astype(np.uint16),
                "cam_right_wrist": depth_right_raw.copy().astype(np.uint16),
            },
        }


if __name__ == "__main__":
    from openpi_client import image_tools

    aaa = ImageRecorder()
    # time.sleep(2)
    show_canvas = np.zeros((480, 640 * 3, 3), dtype=np.uint8)
    while 1:
        bbb = aaa.get_images()
        # print(bbb["cam_high"].shape)
        # ccc = image_tools.resize_with_pad(bbb["cam_high"], 224, 224)
        # print(ccc.shape)
        show_canvas[:, :640] = bbb["cam_high"]
        show_canvas[:, 640:640 * 2] = bbb["cam_left_wrist"]
        show_canvas[:, 640 * 2:640 * 3] = bbb["cam_right_wrist"]
        cv2.imshow("demo", show_canvas)
        cv2.waitKey(1)
