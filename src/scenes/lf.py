# FILE: src/scenes/lf.py (REVISED)

import os
import time
import cv2
import numpy as np
from src.actions import SetServo, Start, TurnLeft, TurnRight, Advance
from src.scenes.base_scene import BaseScene
from src.utils import log
from src.models.quickLF import LFModel 

class LF(BaseScene):
    def __init__(self, memory_name, camera_info, msg_queue): 
            super().__init__(memory_name, camera_info, msg_queue)
            self.height = camera_info['height']
            self.width = camera_info['width']
            
            self.vis_window_name = "Lane Keeping View (Cropped)"
            cv2.namedWindow(self.vis_window_name, cv2.WINDOW_NORMAL)
            display_width, display_height = 1024, int(1024 * (self.height * 2 / 3) / self.width)
            cv2.resizeWindow(self.vis_window_name, display_width, display_height)

    def init_state(self):
        #model_path="/root/Atalas/Car/LLM_源码/src/models/modelS_0.om"
        model_path="/home/HwHiAiUser/E2E-Sample/Car/LLM_final/weights/lanenet_50.om"
        log.info(f'start init --- {self.__class__.__name__}')
        self.model = LFModel(model_path)
        log.info(f'{self.__class__.__name__} --- model init succ.')
        self.ctrl.execute(SetServo(servo=[90, 65]))
        return False
    
    # --- HELPER METHODS ---
    
    def _transform_model_to_cropped_coords(self, lane_results_model, model_dims, cropped_dims):
        """
        Transforms lane parameters (k, b) and passes through the confidence score.
        Input: list of (k_model, b_model, confidence)
        Output: list of (k_new, b_new, confidence)
        """
        model_w, model_h = model_dims
        cropped_w, cropped_h = cropped_dims
        scale_x, scale_y = cropped_w / model_w, cropped_h / model_h
        
        lanes_in_cropped_coords = []
        if scale_y > 1e-6:
            for k_model, b_model, confidence in lane_results_model:
                k_new = k_model * (scale_x / scale_y)
                b_new = b_model * scale_x
                lanes_in_cropped_coords.append((k_new, b_new, confidence))
        return lanes_in_cropped_coords

    def _compute_steering_command(self, lane_params, image_width, image_height):
        """Input is a list of (k, b, confidence) tuples."""
        Kp = 0.1
        if len(lane_params) < 2:
            return -0.1
        
        ks = [k for k, b, _ in lane_params]
        all_positive = all(k > 0 for k in ks)
        all_negative = all(k < 0 for k in ks)
        if all_positive or all_negative:
            return -0.1
        # We only need k and b for geometry, so we unpack them
        stable_lanes_kb = [(k, b) for k, b, _ in lane_params]

        y_bottom = image_height - 1
        lanes_with_x = [(p, p[0] * y_bottom + p[1]) for p in stable_lanes_kb]
        lanes_with_x.sort(key=lambda item: item[1])
        
        k_left, b_left = lanes_with_x[0][0]
        k_right, b_right = lanes_with_x[-1][0]
        
        position_error = ((k_left * y_bottom + b_left) + (k_right * y_bottom + b_right)) / 2.0 - image_width / 2.0
        steering_command = Kp * position_error
        
        return steering_command

    def _draw_visualization(self, image, lane_params, steering_command):
        """Input is a list of (k, b, confidence) tuples."""
        for k, b, confidence in lane_params:
            y1, y2 = 0, image.shape[0] - 1
            x1, x2 = int(k * y1 + b), int(k * y2 + b)
            retval, pt1, pt2 = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), (x1, y1), (x2, y2))
            if retval:
                cv2.line(image, pt1, pt2, (0, 255, 0), 2)
                # Display confidence score on the line
                text_pos = ( (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2 )
                cv2.putText(image, str(confidence), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        h, w, _ = image.shape
        arrow_start_pt = (w // 2, h - 20)
        angle_rad = np.radians(-steering_command * 4.0)
        arrow_end_x = int(arrow_start_pt[0] + 100 * np.sin(angle_rad))
        arrow_end_y = int(arrow_start_pt[1] - 100 * np.cos(angle_rad))
        cv2.arrowedLine(image, arrow_start_pt, (arrow_end_x, arrow_end_y), (0, 0, 255), 7, line_type=cv2.LINE_AA, tipLength=0.4)

    def loop(self):
        ret = self.init_state()
        if ret: log.error(f'{self.__class__.__name__} init failed.'); return

        frame = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.broadcaster.buf)
        self.ctrl.execute(Start())
        log.info(f'LF loop start - Confidence Filtering Enabled')
        
        # --- TUNING PARAMETER ---
        # The minimum number of pixels a lane must have to be considered valid.
        LANE_CONFIDENCE_THRESHOLD = 0

        try:
            while True:
                if self.stop_sign.value: break
                if self.pause_sign.value: continue

                # 1. CROP IMAGE
                img_cropped = frame[self.height // 3:, :]
                cropped_h, cropped_w, _ = img_cropped.shape
                
                # 2. GET PREDICTIONS: Returns (k, b, confidence) tuples
                lane_results_model, inference_time = self.model.pred(img_cropped)

                # 3. CONFIDENCE FILTERING: Keep only high-confidence lanes
                confident_lanes_model = [lane for lane in lane_results_model if lane[2] > LANE_CONFIDENCE_THRESHOLD]
                
                log.info(f"Detected {len(lane_results_model)} raw lanes, kept {len(confident_lanes_model)} after confidence filtering.")

                # 4. TRANSFORM COORDS of confident lanes
                lanes_in_cropped_coords = self._transform_model_to_cropped_coords(
                    confident_lanes_model, 
                    (self.model.model_width, self.model.model_height), 
                    (cropped_w, cropped_h)
                )

                # 5. DECISION MAKING based on confident lanes
                steering_command = self._compute_steering_command(lanes_in_cropped_coords, cropped_w, cropped_h)

                # 6. VISUALIZATION of confident lanes
                #vis_frame = img_cropped.copy()
                #self._draw_visualization(vis_frame, lanes_in_cropped_coords, steering_command)
                #cv2.imshow(self.vis_window_name, vis_frame)
                #cv2.waitKey(1)

                log.info(f'Steering Command = {steering_command:.2f}')

                LEFT_TURN_THRESHOLD = 0.0
                RIGHT_TURN_THRESHOLD = 3.0
                
                if LEFT_TURN_THRESHOLD < steering_command < RIGHT_TURN_THRESHOLD:
                    self.ctrl.execute(Advance(speed=21,degree=0.0))
                elif steering_command >= RIGHT_TURN_THRESHOLD:
                    if steering_command>=30:
                        self.ctrl.execute(TurnRight(speed=25, degree=0.9))
                    else:
                        self.ctrl.execute(TurnRight(speed=25, degree=0.75))
                    
                else:
                    if steering_command<=-25:
                        self.ctrl.execute(TurnLeft(speed=25, degree=1))
                    elif steering_command<=-40:
                        self.ctrl.execute(TurnLeft(speed=25, degree=1.2))
                    else:
                        self.ctrl.execute(TurnLeft(speed=25, degree=0.75))
                    
                # 8. Get next frame
                frame = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.broadcaster.buf)

        except Exception as e:
            log.error(f'LF loop error: {e}')
        finally:
            cv2.destroyAllWindows()
            log.info("LF loop finished.")