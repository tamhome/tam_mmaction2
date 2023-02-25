#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil

import cv2
import copy as cp
import mmcv
import numpy as np
import torch
# from mmcv import DictAction
from mmcv.runner import load_checkpoint
from mmaction.models import build_detector, build_model, build_recognizer

from mmaction.apis import inference_recognizer, init_recognizer
from mmdeploy.apis import build_task_processor
from mmdeploy.utils.config_utils import load_config
import onnxruntime
import onnx
# import copy

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

try:
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model`, '
                      '`init_pose_model`, and `vis_pose_result` form '
                      '`mmpose.apis`. These apis are required in this demo! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

# rosに関連するインポート
# import tamlib
from tamlib.node_template import Node
from hsrlib.utils import description
from tamlib.cv_bridge import CvBridge
import rospy
import roslib

from sensor_msgs.msg import CompressedImage, Image


FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


PLATEBLUE = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
PLATEBLUE = PLATEBLUE.split('-')
PLATEBLUE = [hex2color(h) for h in PLATEBLUE]


class MMActionServer(Node):
    """
    mmaction2を用いたアクション認識パッケージ
    """

    def __init__(self) -> None:
        super().__init__()
        self.tam_cv_bridge = CvBridge()
        self.cv_img = None
        self.hsr_head_img_msg = CompressedImage()
        self.result_action_msg = Image()

        self.frames = []  # 画像をステップ枚数分保存する
        self.pose_results_list = []  # ステップ枚数分の認識結果を保存する
        self.human_detections_list = []  # ステップ枚数分の認識結果を保存する
        self.array_index = 0
        self.action_recog_counter = 1
        self.action_recog_step = 3  # n回に一回だけ認識する

        self.description = description.load_robot_description()
        self.io_path = roslib.packages.get_pkg_dir("tam_mmaction_pkg") + "/io/"
        self.device = "cuda:0"

        self.short_side = 480
        self.img_h = 480
        self.img_w = 640

        # 人物検出（骨格推定なし）に関するパラメータ
        self.det_score_thr = 0.6
        self.det_config = self.io_path + "human_detection/configs/yolox_tiny_8x8_300e_coco.py"
        self.det_checkpoint = self.io_path + "human_detection/pths/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"
        self.det_model = init_detector(self.det_config, self.det_checkpoint, self.device)
        # 量子化モデルを読み込む
        # self.det_deploy_cfg_path = self.io_path + "human_detection/configs/detection_onnxruntime_static.py"
        # self.det_onnx_pth = self.io_path + "human_detection/pths/end2end.onnx"

        # self.det_deploy_cfg, self.det_model_cfg = load_config(self.det_deploy_cfg_path, self.det_config)
        # self.task_processor = build_task_processor(self.det_config, self.det_deploy_cfg, self.device)
        # self.det_model = self.task_processor.init_backend_model(self.det_model)

        assert self.det_model.CLASSES[0] == 'person', ('We require you to use a detector trained on COCO')

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
        # sampler = [x for x in val_pipeline if x['type'] == 'SampleAVAFrames'][0]
        # clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
        # assert clip_len % 2 == 0, 'We would like to have an even clip_len'

        # window_size = clip_len * frame_interval
        # num_frame = self.action_recog_step
        # self.timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2, self.action_recog_step)

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
        self.pub_register("result_pose", "/mmaction2/pose_estimation/image", Image, queue_size=1)
        self.pub_register("result_action", "/mmaction2/action_estimation/image", Image, queue_size=1)
        self.sub_register("hsr_head_img_msg", self.description.topic.head_rgbd.rgb_compressed, queue_size=1, callback_func=self.run)

    def __del__(self):
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
        self.loginfo("pose estimation start")
        d = [dict(bbox=x) for x in list(det_result)]
        pose = inference_top_down_pose_model(self.pose_model, cv_img, d, format='xyxy')[0]
        return pose

    # def rgb_based_action_recognition(self):
    #     rgb_config = mmcv.Config.fromfile(args.rgb_config)
    #     rgb_config.model.backbone.pretrained = None
    #     rgb_model = build_recognizer(
    #         rgb_config.model, test_cfg=rgb_config.get('test_cfg'))
    #     load_checkpoint(rgb_model, args.rgb_checkpoint, map_location='cpu')
    #     rgb_model.cfg = rgb_config
    #     rgb_model.to(args.device)
    #     rgb_model.eval()
    #     action_results = inference_recognizer(
    #         rgb_model, args.video, label_path=args.label_map)
    #     rgb_action_result = action_results[0][0]
    #     label_map = [x.strip() for x in open(args.label_map).readlines()]
    #     return label_map[rgb_action_result]

    def detection_inference(self, cv_img):
        """Detect human boxes given frame paths.

        Args:
            args (argparse.Namespace): The arguments.
            frame_paths (list[str]): The paths of frames to do detection inference.

        Returns:
            list[np.ndarray]: The human detection results.
        """
        # model = init_detector(self.det_config, self.det_checkpoint, self.device)
        # assert model.CLASSES[0] == 'person', ('We require you to use a detector trained on COCO')
        # results = []
        # print('Performing Human Detection for each frame')
        # prog_bar = mmcv.ProgressBar(len(frame_paths))

        # for frame_path in frame_paths:
        #     result = inference_detector(self.det_model, frame_path)
        #     # We only keep human detections with score larger than det_score_thr
        #     result = result[0][result[0][:, 4] >= self.det_score_thr]
        #     results.append(result)
        #     prog_bar.update()

        # rosから取得した画像から
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
        # rgb_stdet_config = mmcv.Config.fromfile(self.rgb_stdet_config)
        # rgb_stdet_config.merge_from_dict(self.cfg_options)

        # val_pipeline = self.rgb_stdet_config.data.val.pipeline
        sampler = [x for x in self.val_pipeline if x['type'] == 'SampleAVAFrames'][0]
        clip_len, frame_interval = sampler['clip_len'], sampler['frame_interval']
        assert clip_len % 2 == 0, 'We would like to have an even clip_len'

        window_size = clip_len * frame_interval
        num_frame = len(frames)
        timestamps = np.arange(window_size // 2, num_frame + 1 - window_size // 2, self.predict_stepsize)

        # # Get img_norm_cfg
        # img_norm_cfg = self.rgb_stdet_config['img_norm_cfg']
        # if 'to_rgb' not in img_norm_cfg and 'to_bgr' in img_norm_cfg:
        #     to_bgr = img_norm_cfg.pop('to_bgr')
        #     img_norm_cfg['to_rgb'] = to_bgr
        # img_norm_cfg['mean'] = np.array(img_norm_cfg['mean'])
        # img_norm_cfg['std'] = np.array(img_norm_cfg['std'])

        # Build STDET model
        # try:
        #     # In our spatiotemporal detection demo, different actions should have
        #     # the same number of bboxes.
        #     self.rgb_stdet_config['model']['test_cfg']['rcnn']['action_thr'] = .0
        # except KeyError:
        #     pass

        # self.rgb_stdet_config.model.backbone.pretrained = None
        # rgb_stdet_model = build_detector(
        #     self.rgb_stdet_config.model, test_cfg=self.rgb_stdet_config.get('test_cfg'))

        # load_checkpoint(rgb_stdet_model, self.rgb_stdet_checkpoint, map_location='cpu')
        # rgb_stdet_model.to(self.device)
        # rgb_stdet_model.eval()

        predictions = []

        self.loginfo('Performing SpatioTemporal Action Detection for each clip')

        try:
            proposal = human_detections[self.predict_stepsize - 1]
        except IndexError as e:
            self.logerr(e)

        if proposal.shape[0] == 0:
            predictions.append(None)
            return None, None

        # start_frame = 5 - (clip_len // 2 - 1) * frame_interval
        # frame_inds = 0 + np.arange(0, window_size, frame_interval)
        # frame_inds = list(frame_inds - 1)

        # cv2.imshow("test", imgs[0].astype(np.uint8))
        # cv2.waitKey(0)
        # imgs = [frames[ind].astype(np.float32) for ind in frame_inds]
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
        # prog_bar.update()

        return timestamps, predictions

    def abbrev(self, name):
        """Get the abbreviation of label name:

        'take (an object) from (a person)' -> 'take ... from ...'
        """
        while name.find('(') != -1:
            st, ed = name.find('('), name.find(')')
            name = name[:st] + '...' + name[ed + 1:]
        return name

    def action_result_visualizer(self, img, annotations, labels, pose_results, plate=PLATEBLUE) -> np.array:
        h, w, _ = img.shape
        scale_ratio = np.array([w, h, w, h])

        annotation = annotations[0]
        img = vis_pose_result(self.pose_model, img, pose_results)
        for ann in annotation:
            self.loginfo("make action recognition result image.")
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
                    cv2.rectangle(img, st, ed, plate[0], 2)
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
                cv2.rectangle(img, diag0, diag1, plate[k + 1], -1)
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

    def queue(self, src, a):
        dst = np.roll(src, -1)
        dst[-1] = a
        return dst

    def run(self, img_msg):

        # fin_add_flag = False  # 配列への保存方法を管理するためのフラグ
        # frame_paths, original_frames = frame_extraction(args.video)
        # num_frame = len(frame_paths)
        # h, w, _ = original_frames[0].shape

        # Get Human detection results and pose results
        # self.loginfo("wait compressedImage message")
        # self.img_msg = rospy.wait_for_message(self.description.topic.head_rgbd.rgb_compressed, CompressedImage)
        # self.img_msg = rospy.wait_for_message("/camera/rgb/image_raw", Image)

        self.loginfo("start human detection")
        self.cv_img = self.tam_cv_bridge.compressed_imgmsg_to_cv2(img_msg)
        human_detections = self.detection_inference(self.cv_img)

        pose_results = None

        # 骨格推定を行う
        if self.is_skeleton_recog:
            pose_results = self.pose_inference(self.cv_img, human_detections)
            vis_pose_img = vis_pose_result(self.pose_model, self.cv_img.copy(), pose_results)
            pose_results_msg = self.tam_cv_bridge.cv2_to_imgmsg(vis_pose_img)
            self.pub.result_pose.publish(pose_results_msg)
        else:
            # FIX ME: この場合も認識結果をpubするように修正
            # vis_pose_img = vis_pose_result(self.pose_model, self.cv_img.copy(), pose_result)
            # pose_results_msg = self.tam_cv_bridge.cv2_to_imgmsg(vis_pose_img)
            # self.pub.result_pose.publish(pose_results_msg)
            pass

        # アクション認識を行う
        if self.is_action_recogniion:
            # resize frames to shortside 256
            new_w, new_h = mmcv.rescale_size((self.img_w, self.img_h), (256, np.Inf))
            new_img = mmcv.imresize(self.cv_img, (new_w, new_h))
            w_ratio, h_ratio = new_w / self.img_w, new_h / self.img_h

            # ステップサイズに到達していない場合は，単純に認識結果を保存して終了する
            if len(self.frames) < self.predict_stepsize:
                self.pose_results_list.append(pose_results)
                self.human_detections_list.append(human_detections)
                self.frames.append(new_img)
                return

            # 規定の枚数溜まっていたら，最新の結果を入れてからアクション認識を行う
            elif len(self.frames) == self.predict_stepsize:
                self.loginfo("start action recognition with FIFO")
                # self.queue(self.pose_results_list, pose_results)
                self.human_detections_list.pop(0)
                self.frames.pop(0)

                self.human_detections_list.append(human_detections)
                self.frames.append(new_img)
                # self.queue(self.human_detections_list, human_detections)
                # self.queue(self.frames, new_img)

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
                self.logtrace('Use skeleton-based SpatioTemporal Action Detection')
                clip_len, frame_interval = 30, 1
                timestamps, stdet_preds = self.skeleton_based_stdet(self.stdet_label_map, human_detections, pose_results, num_frame, clip_len, frame_interval, h, w)
                for i in range(len(human_detections)):
                    det = human_detections[i]
                    det[:, 0:4:2] *= w_ratio
                    det[:, 1:4:2] *= h_ratio
                    human_detections[i] = torch.from_numpy(det[:, :4]).to(self.device)

            # キーポイントベースではない処理
            else:
                temp_human_detections_list = cp.deepcopy(self.human_detections_list)
                self.loginfo('Use rgb-based SpatioTemporal Action Detection')
                for i in range(len(temp_human_detections_list)):
                    det = temp_human_detections_list[i]
                    det[:, 0:4:2] *= w_ratio
                    det[:, 1:4:2] *= h_ratio
                    temp_human_detections_list[i] = torch.from_numpy(det[:, :4]).to(self.device)
                timestamps, stdet_preds = self.rgb_based_stdet(cp.copy(self.frames), self.stdet_label_map, temp_human_detections_list, self.img_w, self.img_h, new_w, new_h, w_ratio, h_ratio)

            # アクション認識の結果がない場合
            if timestamps is None and stdet_preds is None:
                cv_result = self.cv_img

            # アクション認識の結果がある場合
            else:
                stdet_results = []
                timestamps = [self.predict_stepsize]
                for timestamp, prediction in zip(timestamps, stdet_preds):
                    human_detection = temp_human_detections_list[timestamp - 1]
                    stdet_results.append(self.pack_result(human_detection, prediction, new_h, new_w))

                cv_result = self.action_result_visualizer(self.cv_img, stdet_results, stdet_preds, pose_results)

            self.result_action_msg = self.tam_cv_bridge.cv2_to_imgmsg(cv_result)
            self.pub.result_action.publish(self.result_action_msg)

            # temp_human_detections_list = []
            # self.pose_results_list = []
            # self.human_detections_list = []
            # self.frames = []

            return True

            # dense_n = int(self.predict_stepsize / self.output_stepsize)
            # output_timestamps = self.dense_timestamps(timestamps, dense_n)
            # # frames = [cv2.imread(frame_paths[timestamp - 1]) for timestamp in output_timestamps]

            # print('Performing visualization')
            # pose_model = init_pose_model(self.pose_config, self.pose_checkpoint, self.device)

            # # if args.use_skeleton_recog or args.use_skeleton_stdet:
            # #     pose_results = [
            # #         pose_results[timestamp - 1] for timestamp in output_timestamps
            # #     ]
            # action_result = ""
            # vis_frames = self.visualize(self.frames, stdet_results, pose_results, action_result, pose_model)
            # vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=self.output_fps)
            # vid.write_videofile(args.out_filename)

            # tmp_frame_dir = osp.dirname(frame_paths[0])
            # shutil.rmtree(tmp_frame_dir)



if __name__ == '__main__':
    rospy.init_node(os.path.basename(__file__).split(".")[0])
    mas = MMActionServer()
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        rate.sleep()