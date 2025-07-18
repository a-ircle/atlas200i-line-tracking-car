#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time

import numpy as np
from src.actions import SetServo, Stop, TurnLeftInPlace, TurnRightInPlace, TurnAround,Sto, Advance,  Stopobs2
from src.models import YoloV5
from src.scenes.base_scene import BaseScene
from src.utils import log


class Helper(BaseScene):
    def __init__(self, memory_name, camera_info, msg_queue):
        super().__init__(memory_name, camera_info, msg_queue)
        self.det = None
        self.cls = None

    def init_state(self):
        log.info(f'start init {self.__class__.__name__}')
        #det_path = os.path.join(os.getcwd(), 'weights', 'yolo.om')
        det_path = os.path.join(os.getcwd(), 'weights', 'my_yolo_62e.om')
        if not os.path.exists(det_path):
            log.error(f'Cannot find the offline inference model(.om) file needed for {self.__class__.__name__}  scene.')
            return True
        self.det = YoloV5(det_path)
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
        last_action = None
        try:
            while True:
                if self.stop_sign.value:
                    break
                if self.pause_sign.value:
                    continue
                start = time.time()
                img_bgr = frame.copy()
                bboxes = self.det.infer(img_bgr)
                log.info(f'{bboxes}')
                bboxes = sorted(bboxes, key=lambda x: x[5], reverse=True)
                for x1, y1, x2, y2, cate, score in bboxes:
                    log.info("========cate:{}========".format(cate))
                    x, y = (x1 + x2) // 2, (y1 + y2) // 2
                    log.info("--------{}, {}--------".format(x, y))
                    h, w = y2 - y1, x2 - x1
                    log.info(f'det: {cate}')
                    if last_action != cate and len(bboxes) > 1:
                        cate = last_action
                    print("x: ",x)
                    print("y: ",y)
                    if cate == 'left':
                        # if 420 < x < 950 and y >= 300:
                        if   y >= 450:
                            self.ctrl.execute(TurnLeftInPlace())
                            #self.ctrl.execute(Advance(speed=60))
                            time.sleep(1)
                            break

                    if cate == 'right':
                         if y >= 450:
                        #if  y >= 600:
                            self.ctrl.execute(TurnRightInPlace())
                            time.sleep(1.1)
                            break

                    if cate == 'return':
                        if y >= 450:
                            self.ctrl.execute(TurnAround())
                            time.sleep(1)
                            break
                    if cate == 'obs2':
                        if y>= 200:
                            self.ctrl.execute(Stopobs2())
                            time.sleep(1)
                            break
                    if cate == 'obs1':
                        if y>=470:
                            self.ctrl.execute(Sto())
                            time.sleep(1)
                            break
                    last_action = cate
                    break
                log.info(f'infer cost {time.time() - start}')
        except KeyboardInterrupt:
            self.ctrl.execute(Stop())
