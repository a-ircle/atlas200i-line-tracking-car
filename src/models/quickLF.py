# FILE: src/models/quickLF.py (REVISED)

import cv2
import sys
import os
import numpy as np
import time

try:
    parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    sys.path.append(parent_dir)
    from src.models.lfnet import LFNet
except ImportError as e:
    print(f"Failed to import LFNet model handler: {e}")
    sys.exit(1)

class LFModel:
    """
    Inference engine that returns lane parameters along with a confidence score.
    """
    def __init__(self, model_path: str):
        self.model_width, self.model_height = (512, 256)
        try:
            self.model_handler = LFNet(model_path)
            print(f"Successfully loaded OM model: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def pred(self, image: np.ndarray) -> tuple:
        """
        Performs inference and returns raw results with confidence.

        Args:
            image (np.ndarray): Input image of any size.

        Returns:
            tuple: (lane_results, inference_time_ms)
                   - lane_results: A list of (k, b, confidence) tuples.
                                   Confidence is the number of pixels in the lane.
                   - inference_time_ms: The inference time in milliseconds.
        """
        start_time = time.time()
        try:
            om_outputs = self.model_handler.infer(image)
        except Exception as e:
            print(f"An error occurred during model inference: {e}")
            om_outputs = None
        
        inference_time_ms = (time.time() - start_time) * 1000

        if om_outputs is None or len(om_outputs) == 0:
            return [], inference_time_ms

        binary_logits = om_outputs[0]
        binary_mask = np.argmax(binary_logits, axis=1).squeeze(0).astype(np.uint8)
        
        num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        lane_results = []
        if num_labels > 1:
            for i in range(1, num_labels):
                # The area from stats IS the number of pixels, which is our confidence score.
                confidence = stats[i, cv2.CC_STAT_AREA]
                
                # We can do a preliminary filter here for very small noise
                if confidence < 20: 
                    continue
                
                ys, xs = np.where(labels_map == i)
                if len(ys) < 2:
                    continue
                
                A = np.vstack([ys, np.ones(len(ys))]).T
                try:
                    k_model, b_model = np.linalg.lstsq(A, xs, rcond=None)[0]
                    # Append the tuple with k, b, and confidence
                    lane_results.append((k_model, b_model, confidence))
                except np.linalg.LinAlgError:
                    continue
        
        return lane_results, inference_time_ms