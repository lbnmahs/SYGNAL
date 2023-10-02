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