#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

def test_camera(camera_index=2):
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"--------{camera_index}")
        # �����������ܵ�����
        for i in range(5):
            if i != camera_index:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"========{i}")
                    cap.release()
                    return i
        return -1
    
    print(f"======={camera_index}")
    
    # ��ȡһ֡���в���
    ret, frame = cap.read()
    
    if ret:
        print("win")
        # ��ʾͼ��ߴ�
        height, width, _ = frame.shape
        print(f"witdth:{width}x{height}")
    else:
        print("NONE")
    
    # �ͷ�����ͷ��Դ
    cap.release()
    return camera_index

if __name__ == "__main__":
    camera_index = test_camera()
    if camera_index >= 0:
        print(f"++++++++{camera_index}")
    else:
        print("NULL")