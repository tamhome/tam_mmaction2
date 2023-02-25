# アクション認識を行うためのパッケージ

## model zoo

- human detection
  - [R-50-FPN-pytorch-2x-38.4](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth)

- pose estimation
  - [hr-net-w32](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth)

- action recognition
  - [slowfast-temporal-max](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-345618cd.pth)

## 環境構築
- mmpose, mmdetをインストールすると，mmcv-fullのバージョンを落とされるので上げる

```
pip install openmim
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.7.0
mim install mmdet
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.7.0
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.7.0
cd mmaction2
pip install .
pip install numpy --upgrade
pip install moviepy
```

### install ros
```
apt install curl
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
apt update
```

## cuda関連で困ったときの参考書

- https://mmdetection.readthedocs.io/en/v2.9.0/faq.html

## 環境構築ができているかのテスト用

- mm系のすべてのパッケージに，インストールがどんな状況になっているかを可視化するコードがある．
```
python mmaction/utils/collect_env.py
```


## imstall mmdeploy

```
pip install onnxruntime-gpu
pip install openvino-dev
```
