#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tamlib.utils import Logger
import rospy
import numpy as np
from typing import Dict, List, Optional, Tuple

from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from image_geometry import PinholeCameraModel


class MMActionUtils(Logger):
    """
    認識結果を3次元上に起こして可視化するためのクラス
    """

    def __init__(self) -> None:
        """
        keypointの情報などを初期化
        """

        super().__init__(loglevel="INFO")
        self.loginfo("set keypoint information")

        self.mm_keypoint_info = {
            "nose": {"id": 0, "color": [51, 153, 255], "type": 'upper', "swap": None},
            "left_eye": {"id": 1, "color": [51, 153, 255], "type": 'upper', "swap": 'right_eye'},
            "right_eye": {"id": 2, "color": [51, 153, 255], "type": 'upper', "swap": 'left_eye'},
            "left_ear": {"id": 3, "color": [51, 153, 255], "type": 'upper', "swap": 'right_ear'},
            "right_ear": {"id": 4, "color": [51, 153, 255], "type": 'upper', "swap": 'left_ear'},
            "left_shoulder": {"id": 5, "color": [0, 255, 0], "type": 'upper', "swap": 'right_shoulder'},
            "right_shoulder": {"id": 6, "color": [255, 128, 0], "type": 'upper', "swap": 'left_shoulder'},
            "left_elbow": {"id": 7, "color": [0, 255, 0], "type": 'upper', "swap": 'right_elbow'},
            "right_elbow": {"id": 8, "color": [255, 128, 0], "type": 'upper', "swap": 'left_elbow'},
            "left_wrist": {"id": 9, "color": [0, 255, 0], "type": 'upper', "swap": 'right_wrist'},
            "right_wrist": {"id": 10, "color": [255, 128, 0], "type": 'upper', "swap": 'left_wrist'},
            "left_hip": {"id": 11, "color": [0, 255, 0], "type": 'lower', "swap": 'right_hip'},
            "right_hip": {"id": 12, "color": [255, 128, 0], "type": 'lower', "swap": 'left_hip'},
            "left_knee": {"id": 13, "color": [0, 255, 0], "type": 'lower', "swap": 'right_knee'},
            "right_knee": {"id": 14, "color": [255, 128, 0], "type": 'lower', "swap": 'left_knee'},
            "left_ankle": {"id": 15, "color": [0, 255, 0], "type": 'lower', "swap": 'right_ankle'},
            "right_ankle": {"id": 16, "color": [255, 128, 0], "type": 'lower', "swap": 'left_ankle'}
        }

        self.loginfo("set skeleton infromation")
        self.mm_skeleton_info = {
            "left_ankle2left_knee": {"link": ('left_ankle', 'left_knee'), "id": 0, "color": [0, 255, 0]},
            "left_knee2left_hip": {"link": ('left_knee', 'left_hip'), "id": 1, "color": [0, 255, 0]},
            "right_ankle2right_knee": {"link": ('right_ankle', 'right_knee'), "id": 2, "color": [255, 128, 0]},
            "right_knee2right_hip": {"link": ('right_knee', 'right_hip'), "id": 3, "color": [255, 128, 0]},
            "left_hip2right_hip": {"link": ('left_hip', 'right_hip'), "id": 4, "color": [51, 153, 255]},
            "left_shoulder2left_hip": {"link": ('left_shoulder', 'left_hip'), "id": 5, "color": [51, 153, 255]},
            "right_shoulder2right_hip": {"link": ('right_shoulder', 'right_hip'), "id": 6, "color": [51, 153, 255]},
            "left_shoulder2right_shoulder": {"link": ('left_shoulder', 'right_shoulder'), "id": 7, "color": [51, 153, 255]},
            "left_shoulder2left_elbow": {"link": ('left_shoulder', 'left_elbow'), "id": 8, "color": [0, 255, 0]},
            "right_shoulder2right_elbow": {"link": ('right_shoulder', 'right_elbow'), "id": 9, "color": [255, 128, 0]},
            "left_elbow2left_wrist": {"link": ('left_elbow', 'left_wrist'), "id": 10, "color": [0, 255, 0]},
            "right_elbow2right_wrist": {"link": ('right_elbow', 'right_wrist'), "id": 11, "color": [255, 128, 0]},
            "left_eye2right_eye": {"link": ('left_eye', 'right_eye'), "id": 12, "color": [51, 153, 255]},
            "nose2left_eye": {"link": ('nose', 'left_eye'), "id": 13, "color": [51, 153, 255]},
            "nose2right_eye": {"link": ('nose', 'right_eye'), "id": 14, "color": [51, 153, 255]},
            "left_eye2left_ear": {"link": ('left_eye', 'left_ear'), "id": 15, "color": [51, 153, 255]},
            "right_eye2right_ear": {"link": ('right_eye', 'right_ear'), "id": 16, "color": [51, 153, 255]},
            "left_ear2left_shoulder": {"link": ('left_ear', 'left_shoulder'), "id": 17, "color": [51, 153, 255]},
            "right_ear2right_shoulder": {"link": ('right_ear', 'right_shoulder'), "id": 18, "color": [51, 153, 255]}
        }

        self.mm_skeleton_wholebody_info = {

            'left_ankle_to_left_knee':
            dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
            'left_knee_to_left_hip':
            dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
            'right_ankle_to_right_knee':
            dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
            'right_knee_to_right_hip':
            dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
            'left_hip_to_right_hip':
            dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
            'left_shoulder_to_left_hip':
            dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
            'right_shoulder_to_right_hip':
            dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
            'left_shoulder_to_right_shoulder':
            dict(link=('left_shoulder', 'right_shoulder'), id=7, color=[51, 153, 255]),
            'left_shoulder_to_left_elbow':
            dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
            'right_shoulder_to_right_elbow':
            dict(link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
            'left_elbow_to_left_wrist':
            dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
            'right_elbow_to_right_wrist':
            dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
            'left_eye_to_right_eye':
            dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
            'nose_to_left_eye':
            dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
            'nose_to_right_eye':
            dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
            'left_eye_to_left_ear':
            dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
            'right_eye_to_right_ear':
            dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
            'left_ear_to_left_shoulder':
            dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
            'right_ear_to_right_shoulder':
            dict(link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255]),
            'left_ankle_to_left_big_toe':
            dict(link=('left_ankle', 'left_big_toe'), id=19, color=[0, 255, 0]),
            'left_ankle_to_left_small_toe':
            dict(link=('left_ankle', 'left_small_toe'), id=20, color=[0, 255, 0]),
            'left_ankle_to_left_heel':
            dict(link=('left_ankle', 'left_heel'), id=21, color=[0, 255, 0]),
            'right_ankle_to_right_big_toe':
            dict(link=('right_ankle', 'right_big_toe'), id=22, color=[255, 128, 0]),
            'right_ankle_to_right_small_toe':
            dict(link=('right_ankle', 'right_small_toe'), id=23, color=[255, 128, 0]),
            'right_ankle_to_right_heel':
            dict(link=('right_ankle', 'right_heel'), id=24, color=[255, 128, 0]),
            'left_hand_root_to_left_thumb1':
            dict(link=('left_hand_root', 'left_thumb1'), id=25, color=[255, 128, 0]),
            'left_thumb1_to_left_thumb2':
            dict(link=('left_thumb1', 'left_thumb2'), id=26, color=[255, 128, 0]),
            'left_thumb2_to_left_thumb3':
            dict(link=('left_thumb2', 'left_thumb3'), id=27, color=[255, 128, 0]),
            'left_thumb3_to_left_thumb4':
            dict(link=('left_thumb3', 'left_thumb4'), id=28, color=[255, 128, 0]),
            'left_hand_root_to_left_forefinger1':
            dict(link=('left_hand_root', 'left_forefinger1'), id=29, color=[255, 153, 255]),
            'left_forefinger1_to_left_forefinger2':
            dict(link=('left_forefinger1', 'left_forefinger2'), id=30, color=[255, 153, 255]),
            'left_forefinger2_to_left_forefinger3':
            dict(link=('left_forefinger2', 'left_forefinger3'), id=31, color=[255, 153, 255]),
            'left_forefinger3_to_left_forefinger4':
            dict(link=('left_forefinger3', 'left_forefinger4'), id=32, color=[255, 153, 255]),
            'left_hand_root_to_left_middle_finger1':
            dict(link=('left_hand_root', 'left_middle_finger1'), id=33, color=[102, 178, 255]),
            'left_middle_finger1_to_left_middle_finger2':
            dict(link=('left_middle_finger1', 'left_middle_finger2'), id=34, color=[102, 178, 255]),
            'left_middle_finger2_to_left_middle_finger3':
            dict(link=('left_middle_finger2', 'left_middle_finger3'), id=35, color=[102, 178, 255]),
            'left_middle_finger3_to_left_middle_finger4':
            dict(link=('left_middle_finger3', 'left_middle_finger4'), id=36, color=[102, 178, 255]),
            'left_hand_root_to_left_ring_finger1':
            dict(link=('left_hand_root', 'left_ring_finger1'), id=37, color=[255, 51, 51]),
            'left_ring_finger1_to_left_ring_finger2':
            dict(link=('left_ring_finger1', 'left_ring_finger2'), id=38, color=[255, 51, 51]),
            'left_ring_finger2_to_left_ring_finger3':
            dict(link=('left_ring_finger2', 'left_ring_finger3'), id=39, color=[255, 51, 51]),
            'left_ring_finger3_to_left_ring_finger4':
            dict(link=('left_ring_finger3', 'left_ring_finger4'), id=40, color=[255, 51, 51]),
            'left_hand_root_to_left_pinky_finger1':
            dict(link=('left_hand_root', 'left_pinky_finger1'), id=41, color=[0, 255, 0]),
            'left_pinky_finger1_to_left_pinky_finger2':
            dict(link=('left_pinky_finger1', 'left_pinky_finger2'), id=42, color=[0, 255, 0]),
            'left_pinky_finger2_to_left_pinky_finger3':
            dict(link=('left_pinky_finger2', 'left_pinky_finger3'), id=43, color=[0, 255, 0]),
            'left_pinky_finger3_to_left_pinky_finger4':
            dict(link=('left_pinky_finger3', 'left_pinky_finger4'), id=44, color=[0, 255, 0]),
            'right_hand_root_to_right_thumb1':
            dict(link=('right_hand_root', 'right_thumb1'), id=45, color=[255, 128, 0]),
            'right_thumb1_to_right_thumb2':
            dict(link=('right_thumb1', 'right_thumb2'), id=46, color=[255, 128, 0]),
            'right_thumb2_to_right_thumb3':
            dict(link=('right_thumb2', 'right_thumb3'), id=47, color=[255, 128, 0]),
            'right_thumb3_to_right_thumb4':
            dict(link=('right_thumb3', 'right_thumb4'), id=48, color=[255, 128, 0]),
            'right_hand_root_to_right_forefinger1':
            dict(link=('right_hand_root', 'right_forefinger1'), id=49, color=[255, 153, 255]),
            'right_forefinger1_to_right_forefinger2':
            dict(link=('right_forefinger1', 'right_forefinger2'), id=50, color=[255, 153, 255]),
            'right_forefinger2_to_right_forefinger3':
            dict(link=('right_forefinger2', 'right_forefinger3'), id=51, color=[255, 153, 255]),
            'right_forefinger3_to_right_forefinger4':
            dict(link=('right_forefinger3', 'right_forefinger4'), id=52, color=[255, 153, 255]),
            'right_hand_root_to_right_middle_finger1':
            dict(link=('right_hand_root', 'right_middle_finger1'), id=53, color=[102, 178, 255]),
            'right_middle_finger1_to_right_middle_finger2':
            dict(link=('right_middle_finger1', 'right_middle_finger2'), id=54, color=[102, 178, 255]),
            'right_middle_finger2_to_right_middle_finger3':
            dict(link=('right_middle_finger2', 'right_middle_finger3'), id=55, color=[102, 178, 255]),
            'right_middle_finger3_to_right_middle_finger4':
            dict(link=('right_middle_finger3', 'right_middle_finger4'), id=56, color=[102, 178, 255]),
            'right_hand_root_to_right_ring_finger1':
            dict(link=('right_hand_root', 'right_ring_finger1'), id=57, color=[255, 51, 51]),
            'right_ring_finger1_to_right_ring_finger2':
            dict(link=('right_ring_finger1', 'right_ring_finger2'), id=58, color=[255, 51, 51]),
            'right_ring_finger2_to_right_ring_finger3':
            dict(link=('right_ring_finger2', 'right_ring_finger3'), id=59, color=[255, 51, 51]),
            'right_ring_finger3_to_right_ring_finger4':
            dict(link=('right_ring_finger3', 'right_ring_finger4'), id=60, color=[255, 51, 51]),
            'right_hand_root_to_right_pinky_finger1':
            dict(link=('right_hand_root', 'right_pinky_finger1'), id=61, color=[0, 255, 0]),
            'right_pinky_finger1_to_right_pinky_finger2':
            dict(link=('right_pinky_finger1', 'right_pinky_finger2'), id=62, color=[0, 255, 0]),
            'right_pinky_finger2_to_right_pinky_finger3':
            dict(link=('right_pinky_finger2', 'right_pinky_finger3'), id=63, color=[0, 255, 0]),
            'right_pinky_finger3_to_right_pinky_finger4':
            dict(link=('right_pinky_finger3', 'right_pinky_finger4'), id=64, color=[0, 255, 0])
        }

        self.mm_keypoint_wholebody_info = {
            "nose": {"id": 0, "color": [51, 153, 255], "type": 'upper', "swap": None},
            "left_eye": {"id": 1, "color": [51, 153, 255], "type": 'upper', "swap": 'right_eye'},
            "right_eye": {"id": 2, "color": [51, 153, 255], "type": 'upper', "swap": 'left_eye'},
            "left_ear": {"id": 3, "color": [51, 153, 255], "type": 'upper', "swap": 'right_ear'},
            "right_ear": {"id": 4, "color": [51, 153, 255], "type": 'upper', "swap": 'left_ear'},
            "left_shoulder": {"id": 5, "color": [0, 255, 0], "type": 'upper', "swap": 'right_shoulder'},
            "right_shoulder": {"id": 6, "color": [255, 128, 0], "type": 'upper', "swap": 'left_shoulder'},
            "left_elbow": {"id": 7, "color": [0, 255, 0], "type": 'upper', "swap": 'right_elbow'},
            "right_elbow": {"id": 8, "color": [255, 128, 0], "type": 'upper', "swap": 'left_elbow'},
            "left_wrist": {"id": 9, "color": [0, 255, 0], "type": 'upper', "swap": 'right_wrist'},
            "right_wrist": {"id": 10, "color": [255, 128, 0], "type": 'upper', "swap": 'left_wrist'},
            "left_hip": {"id": 11, "color": [0, 255, 0], "type": 'lower', "swap": 'right_hip'},
            "right_hip": {"id": 12, "color": [255, 128, 0], "type": 'lower', "swap": 'left_hip'},
            "left_knee": {"id": 13, "color": [0, 255, 0], "type": 'lower', "swap": 'right_knee'},
            "right_knee": {"id": 14, "color": [255, 128, 0], "type": 'lower', "swap": 'left_knee'},
            "left_ankle": {"id": 15, "color": [0, 255, 0], "type": 'lower', "swap": 'right_ankle'},
            "right_ankle": {"id": 16, "color": [255, 128, 0], "type": 'lower', "swap": 'left_ankle'},
            'left_big_toe': {"id": 17, "color": [255, 128, 0], "type": 'lower', "swap": 'right_big_toe'},
            'left_small_toe': {"id": 18, "color": [255, 128, 0], "type": 'lower', "swap": 'right_small_toe'},
            'left_heel': {"id": 19, "color": [255, 128, 0], "type": 'lower', "swap": 'right_heel'},
            'right_big_toe': {"id": 20, "color": [255, 128, 0], "type": 'lower', "swap": 'left_big_toe'},
            'right_small_toe': {"id": 21, "color": [255, 128, 0], "type": 'lower', "swap": 'left_small_toe'},
            'right_heel': {"id": 22, "color": [255, 128, 0], "type": 'lower', "swap": 'left_heel'},
            'face-0': {"id": 23, "color": [255, 255, 255], "type": '', "swap": 'face-16'},
            'face-1': {"id": 24, "color": [255, 255, 255], "type": '', "swap": 'face-15'},
            'face-2': {"id": 25, "color": [255, 255, 255], "type": '', "swap": 'face-14'},
            'face-3': {"id": 26, "color": [255, 255, 255], "type": '', "swap": 'face-13'},
            'face-4': {"id": 27, "color": [255, 255, 255], "type": '', "swap": 'face-12'},
            'face-5': {"id": 28, "color": [255, 255, 255], "type": '', "swap": 'face-11'},
            'face-6': {"id": 29, "color": [255, 255, 255], "type": '', "swap": 'face-10'},
            'face-7': {"id": 30, "color": [255, 255, 255], "type": '', "swap": 'face-9'},
            'face-8': {"id": 31, "color": [255, 255, 255], "type": '', "swap": ''},
            'face-9': {"id": 32, "color": [255, 255, 255], "type": '', "swap": 'face-7'},
            'face-10': {"id": 33, "color": [255, 255, 255], "type": '', "swap": 'face-6'},
            'face-11': {"id": 34, "color": [255, 255, 255], "type": '', "swap": 'face-5'},
            'face-12': {"id": 35, "color": [255, 255, 255], "type": '', "swap": 'face-4'},
            'face-13': {"id": 36, "color": [255, 255, 255], "type": '', "swap": 'face-3'},
            'face-14': {"id": 37, "color": [255, 255, 255], "type": '', "swap": 'face-2'},
            'face-15': {"id": 38, "color": [255, 255, 255], "type": '', "swap": 'face-1'},
            'face-16': {"id": 39, "color": [255, 255, 255], "type": '', "swap": 'face-0'},
            'face-17': {"id": 40, "color": [255, 255, 255], "type": '', "swap": 'face-26'},
            'face-18': {"id": 41, "color": [255, 255, 255], "type": '', "swap": 'face-25'},
            'face-19': {"id": 42, "color": [255, 255, 255], "type": '', "swap": 'face-24'},
            'face-20': {"id": 43, "color": [255, 255, 255], "type": '', "swap": 'face-23'},
            'face-21': {"id": 44, "color": [255, 255, 255], "type": '', "swap": 'face-22'},
            'face-22': {"id": 45, "color": [255, 255, 255], "type": '', "swap": 'face-21'},
            'face-23': {"id": 46, "color": [255, 255, 255], "type": '', "swap": 'face-20'},
            'face-24': {"id": 47, "color": [255, 255, 255], "type": '', "swap": 'face-19'},
            'face-25': {"id": 48, "color": [255, 255, 255], "type": '', "swap": 'face-18'},
            'face-26': {"id": 49, "color": [255, 255, 255], "type": '', "swap": 'face-17'},
            'face-27': {"id": 50, "color": [255, 255, 255], "type": '', "swap": ''},
            'face-28': {"id": 51, "color": [255, 255, 255], "type": '', "swap": ''},
            'face-29': {"id": 52, "color": [255, 255, 255], "type": '', "swap": ''},
            'face-30': {"id": 53, "color": [255, 255, 255], "type": '', "swap": ''},
            'face-31': {"id": 54, "color": [255, 255, 255], "type": '', "swap": 'face-35'},
            'face-32': {"id": 55, "color": [255, 255, 255], "type": '', "swap": 'face-34'},
            'face-33': {"id": 56, "color": [255, 255, 255], "type": '', "swap": ''},
            'face-34': {"id": 57, "color": [255, 255, 255], "type": '', "swap": 'face-32'},
            'face-35': {"id": 58, "color": [255, 255, 255], "type": '', "swap": 'face-31'},
            'face-36': {"id": 59, "color": [255, 255, 255], "type": '', "swap": 'face-45'},
            'face-37': {"id": 60, "color": [255, 255, 255], "type": '', "swap": 'face-44'},
            'face-38': {"id": 61, "color": [255, 255, 255], "type": '', "swap": 'face-43'},
            'face-39': {"id": 62, "color": [255, 255, 255], "type": '', "swap": 'face-42'},
            'face-40': {"id": 63, "color": [255, 255, 255], "type": '', "swap": 'face-47'},
            'face-41': {"id": 64, "color": [255, 255, 255], "type": '', "swap": 'face-46'},
            'face-42': {"id": 65, "color": [255, 255, 255], "type": '', "swap": 'face-39'},
            'face-43': {"id": 66, "color": [255, 255, 255], "type": '', "swap": 'face-38'},
            'face-44': {"id": 67, "color": [255, 255, 255], "type": '', "swap": 'face-37'},
            'face-45': {"id": 68, "color": [255, 255, 255], "type": '', "swap": 'face-36'},
            'face-46': {"id": 69, "color": [255, 255, 255], "type": '', "swap": 'face-41'},
            'face-47': {"id": 70, "color": [255, 255, 255], "type": '', "swap": 'face-40'},
            'face-48': {"id": 71, "color": [255, 255, 255], "type": '', "swap": 'face-54'},
            'face-49': {"id": 72, "color": [255, 255, 255], "type": '', "swap": 'face-53'},
            'face-50': {"id": 73, "color": [255, 255, 255], "type": '', "swap": 'face-52'},
            'face-51': {"id": 74, "color": [255, 255, 255], "type": '', "swap": ''},
            'face-52': {"id": 75, "color": [255, 255, 255], "type": '', "swap": 'face-50'},
            'face-53': {"id": 76, "color": [255, 255, 255], "type": '', "swap": 'face-49'},
            'face-54': {"id": 77, "color": [255, 255, 255], "type": '', "swap": 'face-48'},
            'face-55': {"id": 78, "color": [255, 255, 255], "type": '', "swap": 'face-59'},
            'face-56': {"id": 79, "color": [255, 255, 255], "type": '', "swap": 'face-58'},
            'face-57': {"id": 80, "color": [255, 255, 255], "type": '', "swap": ''},
            'face-58': {"id": 81, "color": [255, 255, 255], "type": '', "swap": 'face-56'},
            'face-59': {"id": 82, "color": [255, 255, 255], "type": '', "swap": 'face-55'},
            'face-60': {"id": 83, "color": [255, 255, 255], "type": '', "swap": 'face-64'},
            'face-61': {"id": 84, "color": [255, 255, 255], "type": '', "swap": 'face-63'},
            'face-62': {"id": 85, "color": [255, 255, 255], "type": '', "swap": ''},
            'face-63': {"id": 86, "color": [255, 255, 255], "type": '', "swap": 'face-61'},
            'face-64': {"id": 87, "color": [255, 255, 255], "type": '', "swap": 'face-60'},
            'face-65': {"id": 88, "color": [255, 255, 255], "type": '', "swap": 'face-67'},
            'face-66': {"id": 89, "color": [255, 255, 255], "type": '', "swap": ''},
            'face-67': {"id": 90, "color": [255, 255, 255], "type": '', "swap": 'face-65'},
            'left_hand_root': {"id": 91, "color": [255, 255, 255], "type": '', "swap": 'right_hand_root'},
            'left_thumb1': {"id": 92, "color": [255, 128, 0], "type": '', "swap": 'right_thumb1'},
            'left_thumb2': {"id": 93, "color": [255, 128, 0], "type": '', "swap": 'right_thumb2'},
            'left_thumb3': {"id": 94, "color": [255, 128, 0], "type": '', "swap": 'right_thumb3'},
            'left_thumb4': {"id": 95, "color": [255, 128, 0], "type": '', "swap": 'right_thumb4'},
            'left_forefinger1': {"id": 96, "color": [255, 153, 255], "type": '', "swap": 'right_forefinger1'},
            'left_forefinger2': {"id": 97, "color": [255, 153, 255], "type": '', "swap": 'right_forefinger2'},
            'left_forefinger3': {"id": 98, "color": [255, 153, 255], "type": '', "swap": 'right_forefinger3'},
            'left_forefinger4': {"id": 99, "color": [255, 153, 255], "type": '', "swap": 'right_forefinger4'},
            'left_middle_finger1': {"id": 100, "color": [102, 178, 255], "type": '', "swap": 'right_middle_finger1'},
            'left_middle_finger2': {"id": 101, "color": [102, 178, 255], "type": '', "swap": 'right_middle_finger2'},
            'left_middle_finger3': {"id": 102, "color": [102, 178, 255], "type": '', "swap": 'right_middle_finger3'},
            'left_middle_finger4': {"id": 103, "color": [102, 178, 255], "type": '', "swap": 'right_middle_finger4'},
            'left_ring_finger1': {"id": 104, "color": [255, 51, 51], "type": '', "swap": 'right_ring_finger1'},
            'left_ring_finger2': {"id": 105, "color": [255, 51, 51], "type": '', "swap": 'right_ring_finger2'},
            'left_ring_finger3': {"id": 106, "color": [255, 51, 51], "type": '', "swap": 'right_ring_finger3'},
            'left_ring_finger4': {"id": 107, "color": [255, 51, 51], "type": '', "swap": 'right_ring_finger4'},
            'left_pinky_finger1': {"id": 108, "color": [0, 255, 0], "type": '', "swap": 'right_pinky_finger1'},
            'left_pinky_finger2': {"id": 109, "color": [0, 255, 0], "type": '', "swap": 'right_pinky_finger2'},
            'left_pinky_finger3': {"id": 110, "color": [0, 255, 0], "type": '', "swap": 'right_pinky_finger3'},
            'left_pinky_finger4': {"id": 111, "color": [0, 255, 0], "type": '', "swap": 'right_pinky_finger4'},
            'right_hand_root': {"id": 112, "color": [255, 255, 255], "type": '', "swap": 'left_hand_root'},
            'right_thumb1': {"id": 113, "color": [255, 128, 0], "type": '', "swap": 'left_thumb1'},
            'right_thumb2': {"id": 114, "color": [255, 128, 0], "type": '', "swap": 'left_thumb2'},
            'right_thumb3': {"id": 115, "color": [255, 128, 0], "type": '', "swap": 'left_thumb3'},
            'right_thumb4': {"id": 116, "color": [255, 128, 0], "type": '', "swap": 'left_thumb4'},
            'right_forefinger1': {"id": 117, "color": [255, 153, 255], "type": '', "swap": 'left_forefinger1'},
            'right_forefinger2': {"id": 118, "color": [255, 153, 255], "type": '', "swap": 'left_forefinger2'},
            'right_forefinger3': {"id": 119, "color": [255, 153, 255], "type": '', "swap": 'left_forefinger3'},
            'right_forefinger4': {"id": 120, "color": [255, 153, 255], "type": '', "swap": 'left_forefinger4'},
            'right_middle_finger1': {"id": 121, "color": [102, 178, 255], "type": '', "swap": 'left_middle_finger1'},
            'right_middle_finger2': {"id": 122, "color": [102, 178, 255], "type": '', "swap": 'left_middle_finger2'},
            'right_middle_finger3': {"id": 123, "color": [102, 178, 255], "type": '', "swap": 'left_middle_finger3'},
            'right_middle_finger4': {"id": 124, "color": [102, 178, 255], "type": '', "swap": 'left_middle_finger4'},
            'right_ring_finger1': {"id": 125, "color": [255, 51, 51], "type": '', "swap": 'left_ring_finger1'},
            'right_ring_finger2': {"id": 126, "color": [255, 51, 51], "type": '', "swap": 'left_ring_finger2'},
            'right_ring_finger3': {"id": 127, "color": [255, 51, 51], "type": '', "swap": 'left_ring_finger3'},
            'right_ring_finger4': {"id": 128, "color": [255, 51, 51], "type": '', "swap": 'left_ring_finger4'},
            'right_pinky_finger1': {"id": 129, "color": [0, 255, 0], "type": '', "swap": 'left_pinky_finger1'},
            'right_pinky_finger2': {"id": 130, "color": [0, 255, 0], "type": '', "swap": 'left_pinky_finger2'},
            'right_pinky_finger3': {"id": 131, "color": [0, 255, 0], "type": '', "swap": 'left_pinky_finger3'},
            'right_pinky_finger4': {"id": 132, "color": [0, 255, 0], "type": '', "swap": 'left_pinky_finger4'}
        }

        use_wholebody = True
        if use_wholebody:
            self.mm_keypoint_info = self.mm_keypoint_wholebody_info
            self.mm_skeleton_info = self.mm_skeleton_wholebody_info

        self.pose_th = 0.6

        self._p_offset_x = 0.0
        self._p_offset_y = 0.0
        self._p_offset_z = 0.0

        self.logsuccess("MMActionUtilsの初期化完了")
        return

    def pixelTo3D(self, key_point: np.ndarray, cv_d: np.ndarray, camera_model: PinholeCameraModel) -> Dict[Point, float]:
        """
        (x, y)で表される2次元上の画像座標を3次元座標に変換する関数
        Args:
            key_point(np.ndarray): キーポイントの画像上の2点 [x, y],
            cv_d(np.ndarray): depth image
            camera_model(PinholeCameraModel): 使用しているカメラinfoから算出したカメラモデル
                fromCameraInfo(camera_info_msg)
        Returns:
            Tuple (geometry_msgs.msg.Point型の座標情報, 信頼値)
            計算に失敗した場合はFalseを返す
        """

        point_x, point_y = key_point[0], key_point[1]
        confidence = key_point[2]
        self.logdebug("point_x: " + str(point_x))
        self.logdebug("point_y: " + str(point_y))
        self.logdebug("confidence: " + str(key_point[2]))

        # キーポイントが入っていない場合
        if point_x == 0 and point_y == 0:
            self.logdebug("x and y is 0")
            return False

        # 重心算出
        cx = int(point_x)
        cy = int(point_y)

        # 重心周辺のDepth取得（ノイズ対策）
        kernel = [-1, 0, 1]
        depth_list = []
        for y in kernel:
            for x in kernel:
                try:
                    depth = cv_d[cy + y, cx + x] * 0.001  # mmからmに変更
                    if depth > 0:
                        depth_list.append(depth)
                except Exception as e:
                    self.logdebug(e)
                    continue

        if len(depth_list) != 0:
            # 座標算出
            uv = list(camera_model.projectPixelTo3dRay((cx, cy)))
            uv[:] = [x / uv[2] for x in uv]
            uv[:] = [x * np.mean(depth_list) for x in uv]
        else:
            self.logdebug("depth_list length is 0")
            return False

        return {"point": Point(uv[0] + self._p_offset_x, uv[1] + self._p_offset_y, uv[2] + self._p_offset_z), "conf": confidence}

    def createMarker(self, frame_id, ns, marker_id, keypoint_3d, person, point1, point2) -> Marker:
        """
        2つのポイントから描画用のマーカーを作成する関数
        """

        # 信頼値情報から描画するかどうかを選択
        if keypoint_3d[self.mm_keypoint_info[point1]["id"]]["conf"] > self.pose_th and keypoint_3d[self.mm_keypoint_info[point2]["id"]]["conf"] > self.pose_th:
            color = self.mm_skeleton_info[ns]["color"]

            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.lifetime = rospy.Duration(0.5)
            marker.ns = ns
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.color.r = color[2] / 255.0
            marker.color.g = color[1] / 255.0
            marker.color.b = color[0] / 255.0
            marker.color.a = 1.0
            marker.scale.x = 0.03
            # marker.pose.orientation.w = 1.0
            marker.points.append(keypoint_3d[self.mm_keypoint_info[point1]["id"]]["point"])
            marker.points.append(keypoint_3d[self.mm_keypoint_info[point2]["id"]]["point"])

            return marker

        else:
            return False

    def display3DPose(self, people_keypoints_3d, frame: str) -> MarkerArray:
        """
        複数人物のキーポイント情報をMarkerArrayを使って可視化する関数
        """
        marker_array = MarkerArray()
        for i, keypoints_3d in enumerate(people_keypoints_3d):
            if keypoints_3d is not None:
                for key, value in self.mm_skeleton_info.items():
                    if keypoints_3d[self.mm_keypoint_info[value["link"][0]]["id"]] and keypoints_3d[self.mm_keypoint_info[value["link"][1]]["id"]]:
                        marker = self.createMarker(
                            frame,
                            key,
                            (i * 20) + value["id"],
                            keypoints_3d,
                            None,
                            value["link"][0],
                            value["link"][1]
                        )
                        if marker:
                            marker_array.markers.append(marker)
        return marker_array


if __name__ == "__main__":
    mmaction_utils = MMActionUtils()