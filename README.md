# アクション認識を行うためのパッケージ

## model zoo

- **太字：現在使用中のモデル**

- [human detectionのモデル配置場所](./io/human_detection/pths/)
- [pose estimationのモデル配置場所](./io/pose_estimation/pths/)
- [action recognitionのモデル配置場所](./io/action_recognition/pths/)

### human detection
- [R-50-FPN-pytorch-2x-38.4](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth)
- [**yolox-tiny**](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth)

### pose estimation
- [hr-net-w32](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth)
- [**seresnet-50**](https://download.openmmlab.com/mmpose/top_down/seresnet/seresnet50_coco_256x192-25058b66_20200727.pth)

### action recognition
- [slowfast-omnisource-r101](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth)
- [**slowfast-temporal-max**](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-345618cd.pth)

## launch

- 認識の最大距離を指定可能
  - メートル単位で指定
  - 0を指定した場合は，depthによるマスク処理を行わない

```xml
    <!-- Hyper parameter -->
    <arg name="max_distance" default="2.0"/>
    <!-- launch action recognition -->
    <node pkg="tam_mmaction2" type="action_recognition_server.py" name="action_recognition_server" output="screen">
        <param name="/max_distance" type="float" value="$(arg max_distance)"/>
    </node>
```

## 環境構築

- mmpose, mmdetをインストールすると，mmcv-fullのバージョンを落とされることがある(?)
  - その場合はmmcv-full==1.7.0を再度インストールする必要あり
  - 公式にしたがって，`mim install mm...`を使用すると， `mmcv-full`のバージョンが下げられるが，`pip install mm...`を使うと問題ない
  - 同様に，`mim install mm...`を使用すると`numpy`のバージョンを下げられることがあるので，その都度`pip install numpy --upgrade`で最新版にする

```bash
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.7.0
pip install mmdet==2.28.1
pip install mmpose==0.29.0
pip install mmaction2==0.24.1
```

### install ros

```bash
apt install curl
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
apt update
```

## cuda関連で困ったときの参考書

- <https://mmdetection.readthedocs.io/en/v2.9.0/faq.html>

## 環境構築ができているかのテスト用

- mm系のすべてのパッケージに，インストールがどんな状況になっているかを可視化するコードがある．
  - 対応するgitのパッケージを落とす必要あり

```bash
python mmdet/utils/collect_env.py
python mmaction/utils/collect_env.py
```

- 出力
- ![mmdet_result](./assets/setup/Screenshot%20from%202023-02-27%2004-09-03.png)
- 左下の部分の， `MMCV CUDA Compiler: 11.7`のところでバージョンが表示されていればGPU込でインストールされている
  - `Unknown`のような表示の場合は，環境構築に失敗している
  - 下記コマンドを再度実行してから，もう一度`mmdet`などのインストールに挑戦すること
    - `MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.7.0`

## imstall mmdeploy（検証中）

```bash
pip install onnxruntime-gpu
pip install openvino-dev
```
