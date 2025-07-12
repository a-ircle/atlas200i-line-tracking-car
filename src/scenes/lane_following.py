import os
import time
import cv2
import numpy as np
from src.actions import SetServo, Stop, Start, TurnLeft, TurnRight, Advance
from src.scenes.base_scene import BaseScene
from src.utils import log

def detect_yellow_line(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 80, 80]) 
    upper_yellow = np.array([35, 255, 255]) 
    
    height, width = hsv.shape[:2]
    bottom_half = hsv[height // 2:, :]  
    mask = cv2.inRange(bottom_half, lower_yellow, upper_yellow)
    # cv2.imshow('Cropped Frame', bottom_half)
    # cv2.waitKey(1)  # 等待1毫秒，以便窗口可以更新
    return mask

def get_line_position(mask):
    height, width = mask.shape
    moments = cv2.moments(mask)
    if moments['m00'] == 0:
        return None
    cx = int(moments['m10'] / moments['m00'])
    return cx

class LF(BaseScene):
    def __init__(self, memory_name, camera_info, msg_queue):
        super().__init__(memory_name, camera_info, msg_queue)
        self.net = None
        #self.forward_spd = 22
        self.forward_spd = 0

    def init_state(self):
        log.info(f'start init {self.__class__.__name__}')
        log.info(f'{self.__class__.__name__} model init succ.')
        self.ctrl.execute(SetServo(servo=[90, 65]))
        return False

    def loop(self):
        ret = self.init_state()
        if ret:
            log.error(f'{self.__class__.__name__} init failed.')
            return
        frame = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.broadcaster.buf)
        log.info(f'{self.__class__.__name__} loop start')
        self.ctrl.execute(Start())
        try:
            while True:
                if self.stop_sign.value:
                    break
                if self.pause_sign.value:
                    continue
                start = time.time()
                img_bgr = frame.copy()
                mask = detect_yellow_line(img_bgr)
                cx = get_line_position(mask)
                if cx is None:
                #   log.info("OOOOOOOOOOOOOOOOOOOOO")
                  cx = 0
                if cx is not None:
                    frame_center = frame.shape[1] / 2
                    error = cx - frame_center
                    log.info(f'error: {error}, cx: {cx}, frame_center: {frame_center}') 
                    if error <  -35:  
                        self.ctrl.execute(TurnLeft(degree=10))  
                    elif -35 <=cx <20 :
                        self.ctrl.execute(TurnRight(degree=10))  
                    else:
                        # self.ctrl.execute(Advance(speed=self.forward_spd)) 
                        pass
                # log.info("AAAAAAAAAAAAcx{}BBBframecenter{}".format(cx,frame_center))
                log.info(f'infer cost {time.time() - start}')
        except KeyboardInterrupt:
            self.ctrl.execute(Stop())
