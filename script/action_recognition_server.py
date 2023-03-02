#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) OpenMMLab. All rights reserved.

import os
import cv2
import sys
import mmcv
import torch
import roslib
import copy as cp
import numpy as np
import image_geometry
from mmcv.runner import load_checkpoint
from mmaction.models import build_detector
from action_recognition_utils import MMActionUtils
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)

# トラッキングについて
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
# from mmtrack.models import build_model
from mmtrack.apis import inference_mot, init_model

# from mmaction.apis import inference_recognizer, init_recognizer
# from mmdeploy.apis import build_task_processor
# from mmdeploy.utils.config_utils import load_config
# import onnxruntime
# import onnx
# import copy

# rosに関連するインポート
# import tamlib
from tamlib.node_template import Node
from hsrlib.utils import description
from tamlib.cv_bridge import CvBridge
import rospy
import roslib

# from std_msgs.msg import Int32
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from tam_mmaction2.msg import Ax3DPose, AxKeyPoint
from tam_mmaction2.msg import Ax3DPoseWithLabel, Ax3DPoseWithLabelArray
# from tam_mmaction2.msg import Ax3DPoseArray, Ax3DPose, AxKeyPoint
# from tam_mmaction2.msg import AxActionRecognition

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1

MM_ACTION_NOSE = 0
MM_ACTION_LEFT_EYE = 1
MM_ACTION_RIGHT_EYE = 2
MM_ACTION_LEFT_EAR = 3
MM_ACTION_RIGHT_EAR = 4
MM_ACTION_LEFT_SHOULDER = 5
MM_ACTION_RIGHT_SHOULDER = 6
MM_ACTION_LEFT_ELBOW = 7
MM_ACTION_RIGHT_ELBOW = 8
MM_ACTION_LEFT_WRIST = 9
MM_ACTION_RIGHT_WRIST = 10
MM_ACTION_LEFT_HIP = 11
MM_ACTION_RIGHT_HIP = 12
MM_ACTION_LEFT_KNEE = 13
MM_ACTION_RIGHT_KNEE = 14
MM_ACTION_LEFT_ANKLE = 15
MM_ACTION_RIGHT_ANKLE = 16


class MMActionServer(Node):
    """
    mmaction2を用いたアクション認識パッケージ
    """

    def __init__(self) -> None:
        super().__init__()

        # rosparam
        self.is_tracking = rospy.get_param(rospy.get_name() + "/tracking", True)
        self.tracking_view = rospy.get_param(rospy.get_name() + "/vis_tracking/detail", False)

        self.mmaction_utils = MMActionUtils()
        self.key_list = self.mmaction_utils.mm_keypoint_info.keys()
        self.frames = []  # 画像をステップ枚数分保存する
        self.pose_results_list = []  # ステップ枚数分の認識結果を保存する
        self.human_detections_list = []  # ステップ枚数分の認識結果を保存する
        self.array_index = 0
        self.action_recog_counter = 1
        self.action_recog_step = 3  # n回に一回だけ認識する


        self.description = description.load_robot_description()
        self.io_path = roslib.packages.get_pkg_dir("tam_mmaction2") + "/io/"
        # self.ailia_base_path = roslib.packages.get_pkg_dir("tam_mmaction2") + "/third_pkgs/ailia-models/"
        self.device = "cuda:0"

        self.short_side = 480
        self.img_h = 480
        self.img_w = 640
        self.frame_num = 0  # トラッキングに関する制御用の変数
        self.fps = 30

        # 人物検出（骨格推定なし）に関するパラメータ
        self.det_score_thr = 0.6
        self.det_config = self.io_path + "human_detection/configs/yolox_tiny_8x8_300e_coco.py"
        self.det_checkpoint = self.io_path + "human_detection/pths/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
        self.det_model = init_detector(self.det_config, self.det_checkpoint, self.device)
        assert self.det_model.CLASSES[0] == 'person', ('We require you to use a detector trained on COCO')

        # トラッキングについて
        self.tracking_thr = 0.95
        self.track_config = self.io_path + "tracking/configs/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py"
        self.track_pth = self.io_path + "tracking/pths/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth"
        self.tracking_model = init_model(self.track_config, self.track_pth, device=self.device)

        # 骨格推定に関するパラメータ
        self.pose_config = self.io_path + "pose_estimation/configs/seresnet50_coco_256x192.py"
        self.pose_checkpoint = self.io_path + "pose_estimation/pths/seresnet50_coco_256x192-25058b66_20200727.pth"
        self.pose_model = init_pose_model(self.pose_config, self.pose_checkpoint, self.device)
        # 骨格推定を実施するかどうか
        self.is_skeleton_recog = True

        # アクション認識に関するパラメータ
        self.is_action_recogniion = True
        # self.rgb_stdet_checkpoint = self.io_path + "action_recognition/pths/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth"
        # self.rgb_stdet_config = self.io_path + "action_recognition/configs/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py"
        self.rgb_stdet_checkpoint = self.io_path + "action_recognition/pths/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-345618cd.pth"
        self.rgb_stdet_config = self.io_path + "action_recognition/configs/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py"
        self.label_map_path = self.io_path + "action_recognition/configs/label_map_only_wave.txt"
        self.use_skeleton_stdet = False  # アクション認識を骨格推定ベースで行うかを決定する
        self.action_topk = 4
        self.predict_stepsize = 8
        self.output_stepsize = 1
        self.output_fps = 30
        self.action_score_th = 0.02

        self.cfg_options = {}

        # label名などの読み込み
        self.stdet_label_map = self.load_label_map(self.label_map_path)
        self.rgb_stdet_config = mmcv.Config.fromfile(self.rgb_stdet_config)
        self.rgb_stdet_config.merge_from_dict(self.cfg_options)

        try:
            if self.rgb_stdet_config['data']['train']['custom_classes'] is not None:
                self.stdet_label_map = {
                    id + 1: self.stdet_label_map[cls]
                    for id, cls in enumerate(self.rgb_stdet_config['data']['train']['custom_classes'])
                }
        except KeyError:
            pass

        self.val_pipeline = self.rgb_stdet_config.data.val.pipeline

        # Get img_norm_cfg
        self.img_norm_cfg = self.rgb_stdet_config['img_norm_cfg']
        if 'to_rgb' not in self.img_norm_cfg and 'to_bgr' in self.img_norm_cfg:
            to_bgr = self.img_norm_cfg.pop('to_bgr')
            self.img_norm_cfg['to_rgb'] = to_bgr
        self.img_norm_cfg['mean'] = np.array(self.img_norm_cfg['mean'])
        self.img_norm_cfg['std'] = np.array(self.img_norm_cfg['std'])

        # Build STDET model
        try:
            # In our spatiotemporal detection demo, different actions should have
            # the same number of bboxes.
            self.rgb_stdet_config['model']['test_cfg']['rcnn']['action_thr'] = .0
        except KeyError:
            pass

        self.rgb_stdet_config.model.backbone.pretrained = None
        self.rgb_stdet_model = build_detector(self.rgb_stdet_config.model, test_cfg=self.rgb_stdet_config.get('test_cfg'))

        load_checkpoint(self.rgb_stdet_model, self.rgb_stdet_checkpoint, map_location='cpu')
        self.rgb_stdet_model.to(self.device)
        self.rgb_stdet_model.eval()

        # ROSインタフェース
        self.tam_cv_bridge = CvBridge()
        self.camera_info = CameraInfo()
        self.cv_img = None
        self.hsr_head_img_msg = CompressedImage()
        self.hsr_head_depth_msg = CompressedImage()
        self.result_action_msg = Image()

        self.camera_frame = self.description.frame.rgbd

        p_rgb_topic = self.description.topic.head_rgbd.rgb_compressed
        p_depth_topic = self.description.topic.head_rgbd.depth_compressed
        topics = {"hsr_head_img_msg": p_rgb_topic, "hsr_head_depth_msg": p_depth_topic}
        self.sync_sub_register("rgbd", topics, callback_func=self.run)
        # self.sub_register("camera_info", self.description.topic.head_rgbd.camera_info, queue_size=1, callback_func=self.cb_sub_camera_info)

        # カメラインフォを一度だけサブスクライブする
        camera_info_msg = rospy.wait_for_message(self.description.topic.head_rgbd.camera_info, CameraInfo)
        # depthスケールの算出
        # self.fx, self.fy, self.cx, self.cy, self.depth_scale = self.get_depth_scale(camera_info_msg)
        # カメラモデルの作成
        self.cam_model = image_geometry.PinholeCameraModel()
        self.cam_model.fromCameraInfo(camera_info_msg)

        self.pub_register("result_pose", "/mmaction2/pose_estimation/image", Image, queue_size=1)
        self.pub_register("result_action", "/mmaction2/action_estimation/image", Image, queue_size=1)
        self.pub_register("result_skeleton", "/mmaction2/action_estimation/skeleton", MarkerArray, queue_size=1)
        self.pub_register("people_poses_publisher", "/mmaction2/poses/with_label", Ax3DPoseWithLabelArray, queue_size=1)
        self.pub_register("result_tracking", "/mmaction2/human_tracking/image", Image, queue_size=1)
        # self.pub_register("people_poses_publisher", "/mmaction2/poses", Ax3DPoseArray, queue_size=1)
        # self.pub_register("poses_publisher", "/mmaction2/poses", Ax3DPose, queue_size=1)

        # rosparam
        self.max_distance = rospy.get_param(rospy.get_name() + "/max_distance", 0)

    def __del__(self):
        """
        デストラクタ
        """
        self.loginfo("delete: tam_mmaction_recognition")

    def load_label_map(self, file_path):
        """Load Label Map.

        Args:
            file_path (str): The file path of label map.

        Returns:
            dict: The label map (int -> label name).
        """
        self.loginfo(file_path)
        lines = open(file_path).readlines()
        lines = [x.strip().split(': ') for x in lines]
        return {int(x[0]): x[1] for x in lines}

    def pose_inference(self, cv_img, det_result) -> any:
        """
        人の検出結果に基づいた範囲で骨格推定を行う関数
        Args:
            cv_img: Opencv形式の画像
            det_result: 人の検出結果
        Returns:
            骨格推定の情報
        """
        self.logdebug("pose estimation start")
        d = [dict(bbox=x) for x in list(det_result)]
        pose = inference_top_down_pose_model(self.pose_model, cv_img, d, format='xyxy')[0]
        return pose

    def detection_inference(self, cv_img):
        """Detect human boxes given frame paths.

        Args:
            args (argparse.Namespace): The arguments.
            frame_paths (list[str]): The paths of frames to do detection inference.

        Returns:
            list[np.ndarray]: The human detection results.
        """
        # rosから取得した画像から認識を行う
        result = inference_detector(self.det_model, cv_img)
        result = result[0][result[0][:, self.action_topk] >= self.det_score_thr]

        return result

    def dense_timestamps(self, timestamps, n):
        """Make it nx frames."""
        old_frame_interval = (timestamps[1] - timestamps[0])
        start = timestamps[0] - old_frame_interval / n * (n - 1) / 2
        new_frame_inds = np.arange(
            len(timestamps) * n) * old_frame_interval / n + start
        return new_frame_inds.astype(np.int64)

    # def skeleton_based_action_recognition(self, args, pose_results, num_frame, h, w):
    #     """骨格推定ベースのアクション認識
    #     Todo:実装する
    #     """
    #     fake_anno = dict(
    #         frame_dict='',
    #         label=-1,
    #         img_shape=(h, w),
    #         origin_shape=(h, w),
    #         start_index=0,
    #         modality='Pose',
    #         total_frames=num_frame)
    #     num_person = max([len(x) for x in pose_results])

    #     num_keypoint = 17
    #     keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
    #                         dtype=np.float16)
    #     keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
    #                             dtype=np.float16)
    #     for i, poses in enumerate(pose_results):
    #         for j, pose in enumerate(poses):
    #             pose = pose['keypoints']
    #             keypoint[j, i] = pose[:, :2]
    #             keypoint_score[j, i] = pose[:, 2]

    #     fake_anno['keypoint'] = keypoint
    #     fake_anno['keypoint_score'] = keypoint_score

    #     label_map = [x.strip() for x in open(args.label_map).readlines()]
    #     num_class = len(label_map)

    #     skeleton_config = mmcv.Config.fromfile(args.skeleton_config)
    #     skeleton_config.model.cls_head.num_classes = num_class  # for K400 dataset
    #     skeleton_pipeline = Compose(skeleton_config.test_pipeline)
    #     skeleton_imgs = skeleton_pipeline(fake_anno)['imgs'][None]
    #     skeleton_imgs = skeleton_imgs.to(args.device)

    #     # Build skeleton-based recognition model
    #     skeleton_model = build_model(skeleton_config.model)
    #     load_checkpoint(
    #         skeleton_model, args.skeleton_checkpoint, map_location='cpu')
    #     skeleton_model.to(args.device)
    #     skeleton_model.eval()

    #     with torch.no_grad():
    #         output = skeleton_model(return_loss=False, imgs=skeleton_imgs)

    #     action_idx = np.argmax(output)
    #     skeleton_action_result = label_map[
    #         action_idx]  # skeleton-based action result for the whole video
    #     return skeleton_action_result

    def rgb_based_stdet(self, frames, label_map, human_detections, w, h, new_w, new_h, w_ratio, h_ratio):
        sampler = [x for x in self.val_pipeline if x['type'] == 'SampleAVAFrames'][0]
        clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
        assert clip_len % 2 == 0, 'We would like to have an even clip_len'

        window_size = clip_len * frame_interval
        num_frame = len(frames)
        timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2, self.predict_stepsize)

        predictions = []

        self.logdebug('Performing SpatioTemporal Action Detection for each clip')

        try:
            proposal = human_detections[self.predict_stepsize - 1]
        except IndexError as e:
            self.logerr(e)

        if proposal.shape[0] == 0:
            predictions.append(None)
            return None, None

        imgs = [img.astype(np.float32) for img in frames]
        _ = [mmcv.imnormalize_(img, **self.img_norm_cfg) for img in imgs]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to(self.device)

        with torch.no_grad():
            result = self.rgb_stdet_model(return_loss=False, img=[input_tensor], img_metas=[[dict(img_shape=(new_h, new_w))]], proposals=[[proposal]])
            result = result[0]
            prediction = []
            # N proposals
            for i in range(proposal.shape[0]):
                prediction.append([])

            # Perform action score thr
            for i in range(len(result)):  # 80
                if i + 1 not in label_map:
                    continue
                for j in range(proposal.shape[0]):
                    if result[i][j, 4] > self.action_score_th:
                        prediction[j].append((label_map[i + 1], result[i][j, 4]))
            predictions.append(prediction)

        return timestamps, predictions

    def abbrev(self, name):
        """Get the abbreviation of label name:

        'take (an object) from (a person)' -> 'take ... from ...'
        """
        while name.find('(') != -1:
            st, ed = name.find('('), name.find(')')
            name = name[:st] + '...' + name[ed + 1:]
        return name

    def action_result_visualizer(self, img, annotations, pose_results) -> np.array:
        h, w, _ = img.shape
        scale_ratio = np.array([w, h, w, h])

        annotation = annotations[0]
        img = vis_pose_result(self.pose_model, img, pose_results)
        for ann in annotation:
            self.logdebug("make action recognition result image.")
            box = ann[0]
            label = ann[1]
            if not len(label):
                cv2.putText(img, "None result", (0, 0), FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
                continue
            try:
                score = ann[2]
                box = (box * scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                if not pose_results:
                    # cv2.rectangle(img, st, ed, plate[0], 2)
                    cv2.rectangle(img, st, ed, (255, 50, 50), 2)
            except IndexError as e:
                self.logerr(e)

            for k, lb in enumerate(label):
                if k >= self.action_topk:
                    break
                text = self.abbrev(lb)
                text = ': '.join([text, str(score[k])])
                location = (0 + st[0], 18 + k * 18 + st[1])
                textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE, THICKNESS)[0]
                textwidth = textsize[0]
                diag0 = (location[0] + textwidth, location[1] - 14)
                diag1 = (location[0], location[1] + 2)
                cv2.rectangle(img, diag0, diag1, (255 - (k * 20), 50, 50), -1)
                cv2.putText(img, text, location, FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)

        return img

    def pack_result(self, human_detection, result, img_h, img_w):
        """Short summary.

        Args:
            human_detection (np.ndarray): Human detection result.
            result (type): The predicted label of each human proposal.
            img_h (int): The image height.
            img_w (int): The image width.

        Returns:
            tuple: Tuple of human proposal, label name and label score.
        """
        human_detection[:, 0::2] /= img_w
        human_detection[:, 1::2] /= img_h
        results = []
        if result is None:
            return None
        for prop, res in zip(human_detection, result):
            res.sort(key=lambda x: -x[1])
            results.append((prop.data.cpu().numpy(), [x[0] for x in res], [x[1] for x in res]))
        return results

    # def inference_mot(self, model, img, frame_id):
    #     """Inference image(s) with the mot model.
    #     Args:
    #         model (nn.Module): The loaded mot model.
    #         img (str | ndarray): Either image name or loaded image.
    #         frame_id (int): frame id.
    #     Returns:
    #         dict[str : ndarray]: The tracking results.
    #     """
    #     cfg = model.cfg
    #     device = next(model.parameters()).device  # model device
    #     # prepare data
    #     if isinstance(img, np.ndarray):
    #         # directly add img
    #         data = dict(img=img, img_info=dict(frame_id=frame_id), img_prefix=None)
    #         cfg = cfg.copy()
    #         # set loading pipeline type
    #         cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    #     else:
    #         # add information into dict
    #         data = dict(
    #             img_info=dict(filename=img, frame_id=frame_id), img_prefix=None)
    #     # build the data pipeline
    #     test_pipeline = Compose(cfg.data.test.pipeline)
    #     data = test_pipeline(data)
    #     data = collate([data], samples_per_gpu=1)
    #     if next(model.parameters()).is_cuda:
    #         # scatter to specified GPU
    #         data = scatter(data, [device])[0]
    #     else:
    #         for m in model.modules():
    #             assert not isinstance(
    #                 m, RoIPool
    #             ), 'CPU inference with RoIPool is not supported currently.'
    #         # just get the actual data from DataContainer
    #         data['img_metas'] = data['img_metas'][0].data
    #     # forward the model
    #     with torch.no_grad():
    #         result = model(return_loss=False, rescale=True, **data)
    #     return result

    # def pixel_to_point(pixel, depth, fx, fy, cx, cy, depth_scale):
    #     # x = (pixel[0] - cx) * depth / fx / depth_scale
    #     y = (pixel[1] - cy) * depth / fy / depth_scale
    #     z = depth / depth_scale
    #     return x, y, z

    def calc_iou(self, bbox1, bbox2):
        # box1の座標
        x1_1, y1_1, x2_1, y2_1 = bbox1
        # box2の座標
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # box1とbox2の共通領域の座標
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        # box1とbox2の共通領域の面積を計算する
        if x_right < x_left or y_bottom < y_top:
            intersection_area = 0
        else:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # box1とbox2の面積を計算する
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

        # IoUを計算する
        iou = intersection_area / float(box1_area + box2_area - intersection_area)

        return iou

    def make_depth_mask_img(self, rgb_img, depth_img, threshold):
        # depth画像のthreshold以上のピクセルを選択するマスクを作成する
        mask = (depth_img * 0.001 >= threshold) | (depth_img == 0)

        # マスクを使って、選択されたピクセルを黒色に変更する
        rgb_img[mask] = [0, 255, 0]
        return rgb_img

    def run(self, img_msg, depth_msg):
        """
        アクション認識を行う関数
        """

        if self.run_enable is False:
            return

        self.logdebug("start human detection")
        # print(self.is_tracking)

        self.cv_img = self.tam_cv_bridge.compressed_imgmsg_to_cv2(img_msg)
        self.depth_img = self.tam_cv_bridge.compressed_imgmsg_to_depth(depth_msg)
        self.cv_img4action = self.cv_img.copy()

        # max_distanceが0に設定されているときはマスク処理を行わない
        if self.max_distance != 0.0:
            self.cv_img = self.make_depth_mask_img(rgb_img=self.cv_img.copy(), depth_img=self.depth_img.copy(), threshold=self.max_distance)

        # トラッキングをしよする場合
        if self.is_tracking:
            # トラッキング
            self.tracking_result = inference_mot(self.tracking_model, self.cv_img.copy(), frame_id=self.frame_num)

            # # あとで扱いやすい形に整形 + 結果を描画
            human_detections = []
            tracking_id_list = []
            tracking_img = self.cv_img.copy()

            # 人間だけをトラッキングする
            for track_bbox in self.tracking_result["track_bboxes"][0]:
                np_track_bbox = np.array((track_bbox[1], track_bbox[2], track_bbox[3], track_bbox[4], track_bbox[5]))
                tracking_id_list.append(int(track_bbox[0]))
                human_detections.append(np_track_bbox)
                org1 = (int(track_bbox[1]), int(track_bbox[2]))
                org2 = (int(track_bbox[3]), int(track_bbox[4]))
                cv2.rectangle(tracking_img, org1, org2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_4)
                cv2.putText(tracking_img, str(int(track_bbox[0])), org=org1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_4)

            # トラッキングの結果をpublish
            tracking_img_msg = self.tam_cv_bridge.cv2_to_imgmsg(tracking_img)
            self.pub.result_tracking.publish(tracking_img_msg)

            # float型のnp.arrayになおし，フレームのカウント数を増やす
            human_detections = np.array(human_detections).astype(np.float32)
            # フレーム数のカウントを増やす
            self.frame_num += 1

            if self.frame_num > 100000:
                # フレーム数が一定上になったらリセットする
                self.frame_num = 0

            # self.tracking_model.show_result(
            #     self.cv_img,
            #     self.tracking_result,
            #     score_thr=self.tracking_thr,
            #     show=True,
            #     wait_time=int(1000. / self.fps) if self.fps else 0,
            #     out_file="temp.png",
            #     backend="cv2"
            # )

            if len(self.tracking_result["track_bboxes"][0]) == 0:
                # 人がいなかったときはダミーのデータをパブリッシュ
                msg_pose3d_with_label_array = Ax3DPoseWithLabelArray()
                msg_pose3d_with_label_array.header.stamp = rospy.Time.now()
                self.pub.people_poses_publisher.publish(msg_pose3d_with_label_array)
                return

        else:
            human_detections = self.detection_inference(self.cv_img)
            if len(human_detections) == 0:
                self.logdebug("no human")
                return

        pose_results = None
        # 骨格推定を行う
        if self.is_skeleton_recog:
            self.logdebug("start keypoint estimation")
            pose_results = self.pose_inference(self.cv_img, human_detections)
            vis_pose_img = vis_pose_result(self.pose_model, self.cv_img.copy(), pose_results)
            pose_results_msg = self.tam_cv_bridge.cv2_to_imgmsg(vis_pose_img)
            self.pub.result_pose.publish(pose_results_msg)
            people_keypoints_3d = []  # 複数人分のキーポイント

            # 人ごとに3次元座標を算出しマーカーに出力する
            array_msg_pose_3d = []  # publish用のデータはあとで作成するため，配列に一時保存する

            for poses in pose_results:
                self.logdebug("キーポイントごとの3次元座標を算出する")
                key_points = poses["keypoints"]
                keypoints_3d = []
                msg_pose_3d = Ax3DPose()  # 1人分の3次元キーポイント座標
                for id, key_point in enumerate(key_points):
                    # 3次元座標算出
                    keypoint_3d = self.mmaction_utils.pixelTo3D(key_point, self.depth_img.copy(), camera_model=self.cam_model)
                    keypoints_3d.append(keypoint_3d)

                    # rosにpublishするようのメッセージを作成
                    msg_keypoint_3d = AxKeyPoint()  # 3次元座標と信頼値を入れる

                    # 3次元座標が算出されたとき
                    if keypoint_3d:
                        msg_keypoint_3d.point = keypoint_3d["point"]
                        msg_keypoint_3d.score = keypoint_3d["conf"]
                    else:
                        # 算出できなかったときは信頼値を-1にする
                        # msg_keypoint_3d.point = Point(0, 0, 0)
                        msg_keypoint_3d.score = -1

                    # キーポイントの3次元座標を適切なメッセージの場所に格納
                    if id == MM_ACTION_NOSE:
                        msg_pose_3d.nose = msg_keypoint_3d
                    elif id == MM_ACTION_LEFT_EYE:
                        msg_pose_3d.left_eye = msg_keypoint_3d
                    elif id == MM_ACTION_RIGHT_EYE:
                        msg_pose_3d.right_eye = msg_keypoint_3d
                    elif id == MM_ACTION_LEFT_EAR:
                        msg_pose_3d.left_ear = msg_keypoint_3d
                    elif id == MM_ACTION_RIGHT_EAR:
                        msg_pose_3d.right_ear = msg_keypoint_3d
                    elif id == MM_ACTION_LEFT_SHOULDER:
                        msg_pose_3d.left_shoulder = msg_keypoint_3d
                    elif id == MM_ACTION_RIGHT_SHOULDER:
                        msg_pose_3d.right_shoulder = msg_keypoint_3d
                    elif id == MM_ACTION_LEFT_ELBOW:
                        msg_pose_3d.left_elbow = msg_keypoint_3d
                    elif id == MM_ACTION_RIGHT_ELBOW:
                        msg_pose_3d.right_elbow = msg_keypoint_3d
                    elif id == MM_ACTION_LEFT_WRIST:
                        msg_pose_3d.left_wrist = msg_keypoint_3d
                    elif id == MM_ACTION_RIGHT_WRIST:
                        msg_pose_3d.right_wrist = msg_keypoint_3d
                    elif id == MM_ACTION_LEFT_HIP:
                        msg_pose_3d.left_hip = msg_keypoint_3d
                    elif id == MM_ACTION_RIGHT_HIP:
                        msg_pose_3d.right_hip = msg_keypoint_3d
                    elif id == MM_ACTION_LEFT_KNEE:
                        msg_pose_3d.left_knee = msg_keypoint_3d
                    elif id == MM_ACTION_RIGHT_KNEE:
                        msg_pose_3d.right_knee = msg_keypoint_3d
                    elif id == MM_ACTION_LEFT_ANKLE:
                        msg_pose_3d.left_ankle = msg_keypoint_3d
                    elif id == MM_ACTION_RIGHT_ANKLE:
                        msg_pose_3d.right_ankle = msg_keypoint_3d

                people_keypoints_3d.append(keypoints_3d)
                array_msg_pose_3d.append(msg_pose_3d)
                # msg_people_keypoints_3d.append(msg_pose_3d)

            # 全員分のキーポイントを算出した後で，複数人のキーポイントをまとめてパブリッシュする
            # msg_ax_pose_3d_array.header.stamp = rospy.Time.now()
            # msg_ax_pose_3d_array.people = msg_people_keypoints_3d
            # self.pub.people_poses_publisher.publish(msg_ax_pose_3d_array)

            marker_array = self.mmaction_utils.display3DPose(people_keypoints_3d, frame=self.camera_frame)
            self.pub.result_skeleton.publish(marker_array)

        else:
            # FIX ME: この場合も認識結果をpubするように修正
            # vis_pose_img = vis_pose_result(self.pose_model, self.cv_img.copy(), pose_result)
            # pose_results_msg = self.tam_cv_bridge.cv2_to_imgmsg(vis_pose_img)
            # self.pub.result_pose.publish(pose_results_msg)
            pass

        # アクション認識を行う
        if self.is_action_recogniion:
            self.logdebug("start action recognition")
            # resize frames to shortside 256
            new_w, new_h = mmcv.rescale_size((self.img_w, self.img_h), (256, np.Inf))
            new_img = mmcv.imresize(self.cv_img4action, (new_w, new_h))
            w_ratio, h_ratio = new_w / self.img_w, new_h / self.img_h

            # ステップサイズに到達していない場合は，単純に認識結果を保存して終了する
            if len(self.frames) < self.predict_stepsize:
                self.pose_results_list.append(pose_results)
                self.human_detections_list.append(human_detections)
                self.frames.append(new_img)
                return

            # 規定の枚数溜まっていたら，最新の結果を入れてからアクション認識を行う
            elif len(self.frames) == self.predict_stepsize:
                self.logdebug("start action recognition with FIFO")
                self.human_detections_list.pop(0)
                self.frames.pop(0)

                self.human_detections_list.append(human_detections)
                self.frames.append(new_img)

            # たまりすぎてしまった場合は，一度リセットする
            elif len(self.frames) > self.predict_stepsize:
                self.logwarn("buffer error, Program remove store images.")
                self.pose_results_list = []
                self.human_detections_list = []
                self.frames = []
                return

            # 意図的に動作速度を落とす
            self.action_recog_counter += 1
            if self.action_recog_counter < self.action_recog_step:
                return
            elif self.action_recog_counter > self.action_recog_step:
                self.action_recog_counter = 1
                return

            # ステップごとに，人物が何をしているのかを検出する
            stdet_preds = None

            # キーポイントごとの処理
            if self.use_skeleton_stdet:
                ...
                # FIXME
                # self.logtrace('Use skeleton-based SpatioTemporal Action Detection')
                # clip_len, frame_interval = 30, 1
                # timestamps, stdet_preds = self.skeleton_based_stdet(self.stdet_label_map, human_detections, pose_results, num_frame, clip_len, frame_interval, h, w)
                # for i in range(len(human_detections)):
                #     det = human_detections[i]
                #     det[:, 0:4:2] *= w_ratio
                #     det[:, 1:4:2] *= h_ratio
                #     human_detections[i] = torch.from_numpy(det[:, :4]).to(self.device)

            # キーポイントベースではない処理
            else:
                temp_human_detections_list = cp.deepcopy(self.human_detections_list)
                self.logdebug('Use rgb-based SpatioTemporal Action Detection')
                try:
                    for i in range(len(temp_human_detections_list)):
                        det = temp_human_detections_list[i]
                        det[:, 0:4:2] *= w_ratio
                        det[:, 1:4:2] *= h_ratio
                        temp_human_detections_list[i] = torch.from_numpy(det[:, :4]).to(self.device)
                except IndexError as e:
                    self.logtrace(e)
                    return
                timestamps, stdet_preds = self.rgb_based_stdet(cp.copy(self.frames), self.stdet_label_map, temp_human_detections_list, self.img_w, self.img_h, new_w, new_h, w_ratio, h_ratio)

            self.logdebug("ラベル付きのアクション認識結果をパブリッシュするための準備")
            msg_pose3d_with_label_array = Ax3DPoseWithLabelArray()
            msg_pose3d_with_label_array.header.stamp = rospy.Time.now()
            people_msg_array = []  # メッセージを集約するための配列

            try:
                for i, cv_human_pos in enumerate(human_detections):
                    hand_wave_id = 0
                    # print(cv_human_pos)
                    # print(stdet_preds[0][i])
                    self.logdebug("一人分のラベル付き認識結果を作成")
                    # rosでpublishするデータを作成
                    msg_pose_3d_with_label = Ax3DPoseWithLabel()
                    msg_pose_3d_with_label.keypoints = array_msg_pose_3d[i]
                    msg_pose_3d_with_label.x = int(cv_human_pos[0])
                    msg_pose_3d_with_label.y = int(cv_human_pos[1])
                    msg_pose_3d_with_label.h = int(cv_human_pos[2] - cv_human_pos[0])
                    msg_pose_3d_with_label.w = int(cv_human_pos[3] - cv_human_pos[1])
                    if self.is_tracking:
                        msg_pose_3d_with_label.id = tracking_id_list[i]
                    msg_pose_3d_with_label.score = [stdet_preds[0][i][hand_wave_id][1]]
                    people_msg_array.append(msg_pose_3d_with_label)

                temp_text = "見つけた人の数: " + str(len(people_msg_array))
                self.logdebug(temp_text)

            except IndexError as e:
                self.loginfo(e)
                self.loginfo("delte current data")

            # publish data with label
            self.logdebug("complete calc all person data")
            msg_pose3d_with_label_array.people = people_msg_array
            msg_pose3d_with_label_array.header.frame_id = self.camera_frame
            self.pub.people_poses_publisher.publish(msg_pose3d_with_label_array)

            # 可視化用画像の作成とpub
            # アクション認識の結果がない場合
            if timestamps is None and stdet_preds is None:
                cv_result = self.cv_img

            # アクション認識の結果がある場合
            else:
                self.logdebug("キーポイントとアクション認識の結果を可視化する")
                stdet_results = []
                timestamps = [self.predict_stepsize]
                for timestamp, prediction in zip(timestamps, stdet_preds):
                    human_detection = temp_human_detections_list[timestamp - 1]
                    stdet_results.append(self.pack_result(human_detection, prediction, new_h, new_w))

                cv_result = self.action_result_visualizer(self.cv_img, stdet_results, pose_results)

            self.result_action_msg = self.tam_cv_bridge.cv2_to_imgmsg(cv_result)
            self.pub.result_action.publish(self.result_action_msg)

            return True


if __name__ == '__main__':
    rospy.init_node(os.path.basename(__file__).split(".")[0])
    mas = MMActionServer()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        rate.sleep()
