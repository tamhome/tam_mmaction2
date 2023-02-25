#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import sys
# import time
# import os
# import math
# import threading
from tamlib.utils import Logger

# from torch import fake_quantize_per_channel_affine

import roslib
import rospy
import actionlib
import numpy as np
import message_filters

from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolResponse
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray


class MMActionUtils(Logger):
    """
    認識結果を3次元上に起こして可視化するためのクラス
    """

    def __init__(self) -> None:
        """
        keypointの情報などを初期化
        """

        super().__init__(loglevel="DEBUG")
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

        self.pose_th = 0.6

        self._p_offset_x = 0.0
        self._p_offset_y = 0.0
        self._p_offset_z = 0.0

    def pixelTo3D(self, key_point, cv_d, camera_model) -> Point:
        """
        (x, y)で表される2次元上の画像座標を3次元座標に変換する関数
        Args:
            point,
            cv_d
        Returns:
            geometry_msgs.msg.Point型の座標情報
            計算に失敗した場合はFalseを返す
        """

        # for key, info in self.mm_keypoint_info.items():
        #     # 各キーポイントの3次元座標を算出
        #     point_x = poses["keypoints"][key][0]
        #     point_y = poses["keypoints"][key][1]
        #     print(point_x, point_y)
        point_x, point_y = key_point[0], key_point[1]
        print(point_x, point_y)

        if point_x == 0 and point_y == 0:
            self.logdebug("x and y is 0")
            return False

        # 重心算出
        # cx = int(cv_d.shape[1] * point_x)
        # cy = int(cv_d.shape[0] * point_y)
        cx = int(point_x)
        cy = int(point_y)
        print(cx, cy)

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
                    print(e)
                    continue

        if len(depth_list) != 0:
            # 座標算出
            uv = list(camera_model.projectPixelTo3dRay((cx, cy)))
            uv[:] = [x / uv[2] for x in uv]
            uv[:] = [x * np.mean(depth_list) for x in uv]
        else:
            self.logdebug("depth_list length is 0")
            return False

        return Point(uv[0] + self._p_offset_x, uv[1] + self._p_offset_y, uv[2] + self._p_offset_z)

    def createMarker(self, frame_id, ns, marker_id, keypoint_3d, person, point1, point2):
        """
        2つのポイントから描画用のマーカーを作成する関数
        """
        # threshold = self.pose_th
        # if person.points[point1].score > threshold and person.points[point2].score > threshold:

        # color = self.hsv2rgb(255 * point1 / ailia.POSE_KEYPOINT_CNT, 255, 255)
        color = [100, 100, 200]

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = rospy.Duration(1)
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
        marker.points.append(keypoint_3d[self.mm_keypoint_info[point1]["id"]])
        marker.points.append(keypoint_3d[self.mm_keypoint_info[point2]["id"]])

        return marker

        # else:
        #     return False

    def display3DPose(self, people_keypoints_3d, frame):
        # pose_key = GP_POSE_KEY.copy()
        marker_array = MarkerArray()
        for i, keypoints_3d in enumerate(people_keypoints_3d):
            if keypoints_3d is not None:
                # keypoint_3d = []
                # for j in range(len(pose_key)):
                #     keypoint_3d.append([])
                # for j in range(len(pose_key)):
                #     # keypoint_3d[pose_key[j]] = self.pixelTo3D(person.points[pose_key[j]], cv_d)
                #     tmp_point = self.pixelTo3D(person.points[pose_key[j]], cv_d)
                #     if tmp_point:
                #         keypoint_3d[pose_key[j]] = Point(tmp_point.x, tmp_point.y, tmp_point.z)

                print(keypoints_3d)

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

                # if keypoint_3d[ailia.POSE_KEYPOINT_SHOULDER_LEFT] and keypoint_3d[ailia.POSE_KEYPOINT_SHOULDER_CENTER]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "shoulder_left2shoulder_center",
                #         (i * 20) + 1,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_SHOULDER_LEFT,
                #         ailia.POSE_KEYPOINT_SHOULDER_CENTER
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)
                # if keypoint_3d[ailia.POSE_KEYPOINT_SHOULDER_RIGHT] and keypoint_3d[ailia.POSE_KEYPOINT_SHOULDER_CENTER]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "shoulder_center2shoulder_center",
                #         (i * 20) + 2,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_SHOULDER_RIGHT,
                #         ailia.POSE_KEYPOINT_SHOULDER_CENTER
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)

                # if keypoint_3d[ailia.POSE_KEYPOINT_SHOULDER_RIGHT] and keypoint_3d[ailia.POSE_KEYPOINT_SHOULDER_CENTER]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "shoulder_center2shoulder_center",
                #         (i * 20) + 3,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_SHOULDER_RIGHT,
                #         ailia.POSE_KEYPOINT_SHOULDER_CENTER
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)

                # if keypoint_3d[ailia.POSE_KEYPOINT_EYE_LEFT] and keypoint_3d[ailia.POSE_KEYPOINT_NOSE]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "eye_left2nose",
                #         (i * 20) + 4,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_EYE_LEFT,
                #         ailia.POSE_KEYPOINT_NOSE
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)
                # if keypoint_3d[ailia.POSE_KEYPOINT_EYE_RIGHT] and keypoint_3d[ailia.POSE_KEYPOINT_NOSE]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "eye_right2nose",
                #         (i * 20) + 5,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_EYE_RIGHT,
                #         ailia.POSE_KEYPOINT_NOSE
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)
                # # if keypoint_3d[ailia.POSE_KEYPOINT_EAR_LEFT] and keypoint_3d[ailia.POSE_KEYPOINT_EYE_LEFT]:
                # #     marker = self.createMarker(
                # #         self._p_frame,
                # #         "ear_left2eye_left",
                # #         (i * 20) + 6,
                # #         keypoint_3d,
                # #         person,
                # #         ailia.POSE_KEYPOINT_EAR_LEFT,
                # #         ailia.POSE_KEYPOINT_EYE_LEFT
                # #     )
                # #     if marker:
                # #         marker_array.markers.append(marker)
                # # if keypoint_3d[ailia.POSE_KEYPOINT_EAR_RIGHT] and keypoint_3d[ailia.POSE_KEYPOINT_EYE_RIGHT]:
                # #     marker = self.createMarker(
                # #         self._p_frame,
                # #         "ear_right2eye_right",
                # #         (i * 20) + 7,
                # #         keypoint_3d,
                # #         person,
                # #         ailia.POSE_KEYPOINT_EAR_RIGHT,
                # #         ailia.POSE_KEYPOINT_EYE_RIGHT
                # #     )
                # #     if marker:
                # #         marker_array.markers.append(marker)

                # if keypoint_3d[ailia.POSE_KEYPOINT_ELBOW_LEFT] and keypoint_3d[ailia.POSE_KEYPOINT_SHOULDER_LEFT]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "elbow_left2shoulder_left",
                #         (i * 20) + 8,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_ELBOW_LEFT,
                #         ailia.POSE_KEYPOINT_SHOULDER_LEFT
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)
                # if keypoint_3d[ailia.POSE_KEYPOINT_ELBOW_RIGHT] and keypoint_3d[ailia.POSE_KEYPOINT_SHOULDER_RIGHT]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "elbow_right2shoulder_right",
                #         (i * 20) + 9,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_ELBOW_RIGHT,
                #         ailia.POSE_KEYPOINT_SHOULDER_RIGHT
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)
                # if keypoint_3d[ailia.POSE_KEYPOINT_WRIST_LEFT] and keypoint_3d[ailia.POSE_KEYPOINT_ELBOW_LEFT]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "wrist_left2elbow_left",
                #         (i * 20) + 10,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_WRIST_LEFT,
                #         ailia.POSE_KEYPOINT_ELBOW_LEFT
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)
                # if keypoint_3d[ailia.POSE_KEYPOINT_WRIST_RIGHT] and keypoint_3d[ailia.POSE_KEYPOINT_ELBOW_RIGHT]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "wrist_right2elbow_right",
                #         (i * 20) + 11,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_WRIST_RIGHT,
                #         ailia.POSE_KEYPOINT_ELBOW_RIGHT
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)

                # if keypoint_3d[ailia.POSE_KEYPOINT_BODY_CENTER] and keypoint_3d[ailia.POSE_KEYPOINT_SHOULDER_CENTER]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "body_center2shoulder_center",
                #         (i * 20) + 12,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_BODY_CENTER,
                #         ailia.POSE_KEYPOINT_SHOULDER_CENTER
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)
                # if keypoint_3d[ailia.POSE_KEYPOINT_HIP_LEFT] and keypoint_3d[ailia.POSE_KEYPOINT_BODY_CENTER]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "hip_left2body_center",
                #         (i * 20) + 13,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_HIP_LEFT,
                #         ailia.POSE_KEYPOINT_BODY_CENTER
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)
                # if keypoint_3d[ailia.POSE_KEYPOINT_HIP_RIGHT] and keypoint_3d[ailia.POSE_KEYPOINT_BODY_CENTER]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "hip_right2body_center",
                #         (i * 20) + 14,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_HIP_RIGHT,
                #         ailia.POSE_KEYPOINT_BODY_CENTER
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)

                # if keypoint_3d[ailia.POSE_KEYPOINT_KNEE_LEFT] and keypoint_3d[ailia.POSE_KEYPOINT_HIP_LEFT]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "knee_left2hip_left",
                #         (i * 20) + 15,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_KNEE_LEFT,
                #         ailia.POSE_KEYPOINT_HIP_LEFT
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)
                # if keypoint_3d[ailia.POSE_KEYPOINT_ANKLE_LEFT] and keypoint_3d[ailia.POSE_KEYPOINT_KNEE_LEFT]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "ankle_left2knee_left",
                #         (i * 20) + 16,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_ANKLE_LEFT,
                #         ailia.POSE_KEYPOINT_KNEE_LEFT
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)
                # if keypoint_3d[ailia.POSE_KEYPOINT_KNEE_RIGHT] and keypoint_3d[ailia.POSE_KEYPOINT_HIP_RIGHT]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "knee_right2hip_right",
                #         (i * 20) + 17,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_KNEE_RIGHT,
                #         ailia.POSE_KEYPOINT_HIP_RIGHT
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)
                # if keypoint_3d[ailia.POSE_KEYPOINT_ANKLE_RIGHT] and keypoint_3d[ailia.POSE_KEYPOINT_KNEE_RIGHT]:
                #     marker = self.createMarker(
                #         self._p_frame,
                #         "ankle_right2knee_right",
                #         (i * 20) + 18,
                #         keypoint_3d,
                #         person,
                #         ailia.POSE_KEYPOINT_ANKLE_RIGHT,
                #         ailia.POSE_KEYPOINT_KNEE_RIGHT
                #     )
                #     if marker:
                #         marker_array.markers.append(marker)

        # self._pub_skeleton.publish(marker_array)

        return marker_array

    def run(self):
        ...


if __name__ == "__main__":
    mmaction_utils = MMActionUtils()
    mmaction_utils.run()