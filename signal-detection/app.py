#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model_data import SignalClassifier





# ===== FUNCTIONS ===== #

# ARGUMENTS FUNCTION
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1500)
    parser.add_argument("--height", help='cap height', type=int, default=800)
    parser.add_argument('--use_static_image_mode', action='store_true')

    parser.add_argument(
        "--min_detection_confidence",
        help='min_detection_confidence',
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help='min_tracking_confidence',
        type=int,
        default=0.5
    )

    args = parser.parse_args()

    return args

# MODE SELECTION FUNCTION
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode

# ===== MAIN FUNCTION ===== #
def main():
    # Parsing the arguments
    args = get_args()

    cap_device = args.device+1
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    mode = 0

    # Initializing the camera
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    #  Loading the Mediapipe model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = SignalClassifier()

    # Reading the Labels
    with open(
        'model_data/classifier_data/classifier_labels.csv',
        encoding='utf-8-sig'
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # Initializing the FPS calculator
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Finger Gesture History
    history_length = 16
    finger_gesture_history = deque(maxlen=history_length)



    # ===== INITIALIZE ===== #
if __name__ == '__main__':
    main()