from ctypes import c_bool
from datetime import datetime
from multiprocessing import shared_memory, Value

import cv2
import numpy as np

from src.utils.logger import logger_instance as log


class CameraBroadcaster:
    def __init__(self, camera_info):
        self.height = camera_info.get('height', 480)
        self.width = camera_info.get('width', 640)
        self.fps = camera_info.get('fps', 30)
        self.stop_sign = Value(c_bool, False)
        self.frame = shared_memory.SharedMemory(create=True, size=np.zeros(shape=(self.height, self.width, 3),
                                                                           dtype=np.uint8).nbytes)
        self.memory_name = self.frame.name

    def run(self):
        cap = cv2.VideoCapture()
        try:
          cap.open(0, apiPreference=cv2.CAP_V4L2)
        except:
          cap.open(1, apiPreference=cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Check if the camera settings are correct
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if actual_width != self.width or actual_height != self.height:
            log.error(f'Failed to set the camera resolution to {self.width}x{self.height}. '
                      f'Actual resolution is {int(actual_width)}x{int(actual_height)}.')
            self.stop_sign.value = True

        sender = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.frame.buf)

        try:
            while True:
                if self.stop_sign.value:
                    self.frame.close()
                    self.frame.unlink()
                    break
                start = datetime.now()
                ret, frame = cap.read()
                if not ret:
                    log.error('Failed to read frame from camera.')
                    continue
                if frame.shape[0] != self.height or frame.shape[1] != self.width:
                    log.error(f'Frame shape {frame.shape} does not match expected shape '
                              f'({self.height}, {self.width}, 3).')
                    continue
                end1 = datetime.now()
                sender[:] = frame[:]
                end2 = datetime.now()
                log.debug(f'{self.memory_name}  read time: {end1 - start}, copy time: {end2 - end1}')
        except (KeyboardInterrupt, SystemExit):
            log.info('Cam broadcaster closing')
            self.frame.close()
            self.frame.unlink()
        finally:
            cap.release()
            cv2.destroyAllWindows()

