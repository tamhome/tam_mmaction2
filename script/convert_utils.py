from mmcv import Config
from mmdeploy.models import ONNXRuntimeDeployer
from onnxruntime.quantization import quantize_dynamic
import onnx
from mmpose.apis import inference_top_down_pose_model

cfg_path = "../io/pose_estimation/configs/seresnet50_coco_256x192.py"

## 保存先
save_model_path = "../io/pose_estimation/pths/seresnet50_coco_256x192-25058b66_20200727.onnx"

img_path = "/home/hma/Pictures/Webcam/2022-12-27-144929.jpg"

cfg = Config.fromfile(cfg_path)  # モデルの構成ファイルのパス
deployer = ONNXRuntimeDeployer(cfg.model, cfg.trt_options)  # モデルクラスを読み込む
deployer.prepare_deploy()  # モデルを準備する

ort_session = deployer._session._sess  # mmdeployモデルクラス内のonnxruntimeのセッションを取得
quantized_model = quantize_dynamic(ort_session, per_channel=True)  # モデルの量子化

onnx.save(quantized_model, save_model_path)

model = inference_top_down_pose_model(
    save_model_path,  # 変換されたモデルのパス
    cfg_path,  # モデルの構成ファイルのパス
    img_path,
    device='cuda:0' # 使用するデバイスの指定。'cpu'または'cuda'のいずれか。
)

pose_results, img_canvas = model.inference(
    '/path/to/image/or/video',  # 推論する画像またはビデオのパス
    format='image'  # 出力のフォーマット。'image'または'video'のいずれか。
)

print(pose_results)
