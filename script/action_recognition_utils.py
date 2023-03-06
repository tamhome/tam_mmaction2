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