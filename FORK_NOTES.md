# OpenPCDet Fork Notes

这个 fork 保留 OpenPCDet 原仓库主体，只补充面向 labelCloud 3D 框标注数据的自动转换、Docker/Compose 训练入口，以及 custom dataset 的少量适配修复。

## 行为边界

自 fork 以来的改动只服务于 labelCloud 数据接入、容器化运行和 custom dataset 稳定性，不修改 OpenPCDet 原仓库的通用训练默认值。

默认行为：

- 默认模型仍是 `pv_rcnn`
- 默认增强仍按生成配置运行，`AUG_MODE=full`
- 默认 ground-truth database 由 `prepare` / `create_custom_infos` 生成
- 默认从零训练，`PRETRAINED_MODEL=` 为空
- 默认不恢复断点，`CKPT=` 为空
- 默认训练 80 epoch，`EPOCHS=80`
- 默认转换 labelCloud `centroid_abs` 标签

测试/实验开关：

- `AUG_MODE=safe` / `AUG_MODE=none` 只是用于定位小数据、局部扫描和过拟合问题，不代表 fork 默认训练策略
- `SAMPLE_POINTS` 默认 `-1`，不限制每帧点数；只有显式设置为正整数时才在生成的数据/模型配置里插入 `sample_points`
- DSVT IoU 相关开关默认保持原配置；只有显式设为 `false` 时，才生成 no-IoU 的稳定性实验配置
- `PRETRAINED_MODEL` 只在显式填写时启用，用于外部权重初始化
- `CKPT` 只用于恢复本 fork 训练出的 checkpoint，不用于外部预训练微调

## 使用场景

当前目标是港口堆场货物点云拆分：

- 输入：labelCloud 导出的 `_classes.json`、`pointClouds/`、`labels/`
- 标注：3D box，类别从 `_classes.json` 自动读取
- 点云特征：直接使用 `x y z r g b`，无色点云的 RGB 填 0
- 输出：OpenPCDet 3D 检测结果 `pred_boxes` 写入 `predictions.json`，后续在宿主机按需可视化或裁切

这不是点级语义分割或实例分割流程；当前 fork 只做 3D box detection。

## OpenPCDet 里的 SOTA 选择

OpenPCDet 没有一个对所有数据集、所有任务都绝对统一的“SOTA 模型”。按原仓库 model zoo 和支持情况看：

- Waymo 单帧 LiDAR：`DSVT-Voxel`、`DSVT-Pillar`、`VoxelNeXt` 属于更靠前的一批
- Waymo 多帧 LiDAR：`MPPNet 16 frames` 指标更强，但依赖连续帧和更复杂的数据组织
- nuScenes LiDAR-only：`TransFusion-L`、`VoxelNeXt` 是更强路线
- nuScenes 多模态：`BEVFusion` 指标更高，但需要图像输入，不适合纯点云流程
- KITTI/custom 单帧工程落地：`PV-RCNN` / `PV-RCNN++` 仍然是成熟、稳定、容易改 custom dataset 的强基线

对当前港口堆场货物数据，自动化入口支持四个模型：

```text
pv_rcnn
pv_rcnn_plusplus
dsvt_pillar
dsvt_voxel
```

默认仍是 `pv_rcnn`，原因是：

- 你的数据是自定义静态场景，不是 Waymo 风格连续多帧自动驾驶数据
- 目标是稳定得到可裁切的 3D box，不是刷公开榜单
- 230 多个场景的数据量不大，先用成熟两阶段模型更容易判断数据和标注质量
- `PV-RCNN` 在 OpenPCDet custom dataset 上改动少，训练和排错成本低

`PV-RCNN++` 使用原项目的 `PVRCNNPlusPlus`、CenterHead 和 VectorPool 结构改到 custom dataset；`DSVT-Pillar` / `DSVT-Voxel` 使用原项目 Waymo DSVT 结构改到 custom dataset。DSVT 更吃显存和依赖，镜像里额外安装 `torch-scatter`。

## 保留的改动

1. labelCloud 转换脚本

- `tools/convert_labelcloud_to_custom.py`
  - 读取 `_classes.json` 获取类别，不在脚本里写死类别
  - 读取 labelCloud `centroid_abs` 格式导出的 JSON 标签；对象中心点字段按 labelCloud 结构读取 `centroid`
  - labelCloud `centroid_abs` 语义按 `center + length/width/height + rot_z` 处理：`z` 是 box 中心，`length -> dx`，`width -> dy`，`height -> dz`
  - 将 labelCloud 的 `rot_z` 从度转换为 OpenPCDet 使用的 yaw 弧度 `[-pi, pi]`，不做符号反转
  - 对 x/y 轴旋转做显式报错，因为当前 OpenPCDet 训练配置只支持 yaw-only 3D box
  - 转换点云前先校验 scene id 唯一性，避免同名 PCD/重复 filename 静默覆盖转换产物
  - 读取 `.pcd` 并保存为 `N x 6` 的 `.npy`：`x y z r g b`
  - 按固定随机种子划分训练集 `train` 和评估集 `val`；最终测试/推理点云通过独立 `infer/` 管理
  - 默认增量转换：已存在且未过期的 `.npy` / `.txt` 会跳过；源 PCD、label 或 class 文件更新后自动重写对应产物
  - 默认生成每个模型自己的 dataset cfg：`labelcloud_dataset_<model>.yaml`
  - 默认同时生成四个模型配置：`labelcloud_pv_rcnn.yaml`、`labelcloud_pv_rcnn_plusplus.yaml`、`labelcloud_dsvt_pillar.yaml`、`labelcloud_dsvt_voxel.yaml`

2. 一键训练入口

- `tools/train_labelcloud_pipeline.py`
  - 串联转换、`create_custom_infos`、训练
  - 默认训练配置跟随 `--out-dir` 自动派生
  - 默认 `--num-gpus auto`，使用容器内可见的所有 GPU；多卡时通过 PyTorch 1.13 兼容的 `torch.distributed.run` 启动 OpenPCDet 分布式训练
  - 默认 `--batch-size auto`，按每张 GPU 2 个样本自动计算总 batch，避免多卡时 batch 不能整除 GPU 数
  - 支持 `--model pv_rcnn|pv_rcnn_plusplus|dsvt_pillar|dsvt_voxel`
  - 支持 `--aug-mode full|safe|none`，默认 `full`
  - 支持 `--sample-points` / `SAMPLE_POINTS`，用于在生成的数据/模型配置中限制每帧训练点数
  - 支持 `DSVT_IOU_HEAD`、`DSVT_IOU_REG_LOSS`、`DSVT_USE_IOU_TO_RECTIFY_SCORE`，用于复现 DSVT no-IoU 稳定性配置
  - 支持 `--pretrained-model` / `PRETRAINED_MODEL`，只加载能匹配的模型参数，不恢复 optimizer
  - 支持 `--ckpt` / `CKPT`，用于恢复本项目训练断点，会恢复 optimizer
  - 支持 `--skip-convert`、`--skip-info` 和 `--skip-train`，用于分阶段转换、准备数据和复用已准备数据重新训练

3. custom dataset 适配修复

- `pcdet/datasets/custom/custom_dataset.py`
  - 支持空标签文件，返回空 GT box
  - ground-truth database 以二进制方式写入 `.bin`
  - ground-truth database 默认可续跑：已完成且未过期的单目标 `.bin` 会跳过，失败后重新执行 `prepare` 会从缺失/过期对象继续
  - ground-truth database 按当前场景缺失 box 数自动计算 `points_in_boxes_cpu` 分块大小，避免对超大场景一次性计算导致崩溃
  - `create_custom_infos` 从 dataset config 读取 `CLASS_NAMES`，不再固定为 `Vehicle/Pedestrian/Cyclist`
  - 点特征数量从 dataset config 读取，支持 `x y z r g b`
- `pcdet/datasets/__init__.py`
  - Argo2 数据集在缺少 `av2` 依赖时改为可选注册，避免 custom 训练被 Python 3.8 不支持的 `av2` 版本阻断；其他导入错误仍会正常暴露
- `pcdet/config.py`
  - `_BASE_CONFIG_` 在当前工作目录解析失败时，会回退到当前 cfg 文件所在目录解析，避免 labelCloud 生成配置在训练和推理入口间路径不一致

4. Docker 自动化

- `docker/Dockerfile.local-cu116`
  - 基于 CUDA 11.6
  - 安装 PyTorch 1.13.1 + cu116、spconv-cu116、torch-scatter、Open3D 和 OpenPCDet 依赖
  - 包含 `opencv-python` / Open3D 常见运行库，避免容器内导入缺共享库
  - 默认使用清华 Ubuntu apt 镜像、阿里 PyPI 镜像，并通过阿里 PyTorch wheels 直链安装 cu116 wheel
  - 基础 CUDA 镜像可通过 `BASE_IMAGE` 覆盖，用于接入 Docker Hub 加速器或内网镜像仓库
  - 复制当前 fork 源码并执行 `python3 setup.py develop`
- `docker-compose.yml`
  - 提供 `pipeline`、`convert`、`reconvert`、`prepare`、`reprepare`、`train`、`retrain`、`infer`、`shell` 服务
  - 数据和输出通过 volume 挂载，不覆盖镜像内已编译源码
  - `labelCloud/`、`infer/`、`checkpoints/`、`data/`、`output/` 都通过 `.env` 可配置
- `.dockerignore`
  - 排除 `data/`、`output/`、权重、点云、日志等大文件

## Compose 用法

准备 labelCloud 数据目录，目录内包含：

```text
_classes.json
pointClouds/
labels/
```

默认从仓库根目录的 `./labelCloud` 读取 labelCloud 数据。把数据组织成：

```text
labelCloud/
  _classes.json
  pointClouds/
  labels/
```

然后执行：

```bash
docker compose build
docker compose run --rm pipeline
```

推荐复制 `.env.example` 后通过 `.env` 管理本机路径和实验参数：

```bash
cp .env.example .env
```

常用 `.env` 项：

```text
LABELCLOUD_DIR=./labelCloud
INFER_DIR=./infer
CHECKPOINTS_DIR=./checkpoints
DATA_DIR=./data
OUTPUT_DIR=./output

MODEL=pv_rcnn
EXTRA_TAG=
CFG_FILE=

OUT_DIR=data/custom
TRAIN_RATIO=0.8
VAL_RATIO=0.2
SPLIT_SEED=42

BATCH_SIZE=auto
WORKERS=4
NUM_GPUS=auto
MASTER_PORT=18888
EPOCHS=80

AUG_MODE=full
SAMPLE_POINTS=-1
DSVT_IOU_HEAD=true
DSVT_IOU_REG_LOSS=true
DSVT_USE_IOU_TO_RECTIFY_SCORE=true
GT_DATABASE_TARGET_ELEMENTS=4000000
GT_DATABASE_MIN_CHUNK_SIZE=50000
GT_DATABASE_MAX_CHUNK_SIZE=500000
GT_DATABASE_MAX_POINTS=0

CKPT=
PRETRAINED_MODEL=
INPUT_DIR=/workspace/infer
SCORE_THRESH=0.3
PRED_OUT_DIR=
```

如果需要切回官方源或换公司内网源，可以覆盖 build args：

```bash
BASE_IMAGE=nvidia/cuda:11.6.2-devel-ubuntu20.04 \
APT_MIRROR=archive.ubuntu.com \
PIP_INDEX_URL=https://pypi.org/simple \
PIP_TRUSTED_HOST=pypi.org \
PYTORCH_WHEEL_BASE=https://download.pytorch.org/whl/cu116 \
PYG_WHEEL_URL=https://data.pyg.org/whl/torch-1.13.1+cu116.html \
docker compose build
```

选择模型：

```bash
MODEL=pv_rcnn docker compose run --rm pipeline
MODEL=pv_rcnn_plusplus docker compose run --rm pipeline
MODEL=dsvt_pillar docker compose run --rm pipeline
MODEL=dsvt_voxel docker compose run --rm pipeline
```

`MODEL` 会影响训练使用的模型配置、训练输出目录、推理默认 checkpoint 查找路径和默认预测输出目录。没有显式设置时等价于 `MODEL=pv_rcnn`。

增强开关：

```text
AUG_MODE=full  保持生成配置里的默认增强，等价于 fork 默认训练行为
AUG_MODE=safe  禁用 gt_sampling、random_world_flip、random_world_rotation
AUG_MODE=none  禁用 gt_sampling、random_world_flip、random_world_rotation、random_world_scaling
```

`safe` 和 `none` 用于定位训练链路、标注语义和小样本过拟合问题。正式训练时，先用 `full` 跑基线，再按训练结果决定是否收紧增强。

`AUG_MODE` 在训练阶段通过 `--set DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST ...` 动态覆盖；修改 `.env` 后直接执行 `train` / `retrain` 会生效。`SAMPLE_POINTS` 和 DSVT IoU 开关属于生成配置内容，修改后仍需要重新执行 `pipeline` 或 `reconvert`。

点数采样：

```text
SAMPLE_POINTS=-1      保留每帧全部点，等价于 fork 默认行为
SAMPLE_POINTS=200000  生成配置时插入 sample_points，每帧训练和测试最多使用 200000 点
```

`SAMPLE_POINTS` 是为超高点数局部扫描准备的显存控制项。它会改变模型看到的输入点数，属于训练质量、速度和显存之间的权衡；300 万到 400 万点/帧的 DSVT 训练通常需要启用。重新执行 `pipeline` 或 `reconvert` 会按当前 `.env` 重新生成配置。PV-RCNN / PV-RCNN++ 通过 dataset cfg 生效；DSVT 通过模型 cfg 生效。

DSVT IoU 稳定性开关：

```text
DSVT_IOU_HEAD=true                       生成 CenterHead 的 iou 预测分支，等价于原 DSVT 配置
DSVT_IOU_REG_LOSS=true                   启用 IoU/DIoU 辅助回归 loss，等价于原 DSVT 配置
DSVT_USE_IOU_TO_RECTIFY_SCORE=true       推理后处理时用预测 IoU 修正 score，等价于原 DSVT 配置
```

小数据或跨数据集微调出现 NaN 时，使用：

```text
DSVT_IOU_HEAD=false
DSVT_IOU_REG_LOSS=false
DSVT_USE_IOU_TO_RECTIFY_SCORE=false
```

这只影响 `dsvt_pillar` / `dsvt_voxel` 生成的 CenterHead 配置。默认值保持原配置，避免 `.env` 变成第二套模型 YAML；该开关只用于复现已验证的数值稳定性分支。`DSVT_IOU_HEAD=false` 时，生成器会同时关闭依赖该分支的 `IOU_REG_LOSS` 和 score rectification，避免生成不一致的 CenterHead 配置。

不同模型的隔离规则：

- 共享：`data/custom/points`、`data/custom/labels`、`data/custom/ImageSets`。这些来自 labelCloud 原始数据，与模型结构无关。
- 独立：`data/custom/cfgs/labelcloud_dataset_<model>.yaml`、`data/custom/cfgs/labelcloud_<model>.yaml`、`data/custom/model_cache/<model>`、`output/data/custom/cfgs/labelcloud_<model>/<extra_tag>`、`output/predictions/<cfg_stem>/<extra_tag>`。
- `EXTRA_TAG=` 为空时使用 OpenPCDet 默认实验名 `default`，避免默认路径出现 `labelcloud_pv_rcnn/labelcloud_pv_rcnn` 这类重复目录。
- `reconvert` 只清理并重写当前模型的 cfg 和 `model_cache/<model>`，然后增量刷新共享转换数据；不会清理或重写其他模型缓存，也不会清理任何训练输出。
- `reprepare` 只重建当前模型的 `model_cache/<model>`；不会清理其他模型缓存或任何训练输出。
- `retrain` 只清理当前 `MODEL + EXTRA_TAG/CFG_FILE` 对应的训练输出；不会清理转换结果、prepare 缓存或其他模型输出；Compose 入口会显式忽略 `.env` 里的 `CKPT`，保证该服务从当前配置重新开始训练。

服务命名约定：

```text
pipeline   convert + prepare + train 一条龙
convert    增量转换 labelCloud 标注数据，生成 train/val
reconvert  清理当前模型 cfg 和 prepare 缓存后重新转换，生成 train/val
prepare    基于已转换数据生成 infos，并按目标级别续跑 gt_database
reprepare  清理当前模型 prepare 缓存后重新准备
train      基于已转换和准备好的数据训练；已有 checkpoint 时继续训练
retrain    清理当前实验训练输出后从头训练
infer      基于用户自行准备的 ./infer 点云推理并写出 predictions.json
```

只转换不训练：

```bash
docker compose run --rm convert
```

`convert` 是增量逻辑。重复执行时会跳过已经生成且未过期的点云 `.npy` 和标签 `.txt`，但会刷新 split、配置和摘要文件。需要强制全量重建转换产物时执行：

```bash
docker compose run --rm reconvert
```

只基于已转换数据生成 OpenPCDet infos 和 ground-truth database：

```bash
docker compose run --rm prepare
```

`prepare` 会在 `data/custom/model_cache/<model>/` 生成当前模型的 `custom_infos_*.pkl` 和 `gt_database/*.bin`。其中 `gt_database/*.bin` 是按目标级别续跑的：已存在、文件大小合法，并且不早于对应 `points/*.npy` 与 `labels/*.txt` 的对象会跳过。

默认分块不是写死 20 万点，而是：

```text
chunk_size = clamp(GT_DATABASE_TARGET_ELEMENTS / 当前场景缺失 box 数,
                   GT_DATABASE_MIN_CHUNK_SIZE,
                   GT_DATABASE_MAX_CHUNK_SIZE)
```

默认值是：

```text
GT_DATABASE_TARGET_ELEMENTS=4000000
GT_DATABASE_MIN_CHUNK_SIZE=50000
GT_DATABASE_MAX_CHUNK_SIZE=500000
GT_DATABASE_MAX_POINTS=0
```

`GT_DATABASE_TARGET_ELEMENTS` 不是点数，而是每次 `points_in_boxes_cpu` 的目标计算量上限，单位约等于 `box 数 x point 数`。默认 `4000000` 是保守经验值：让单次中间矩阵保持在较小内存规模，同时在常见 10 到 20 个 box 的场景下得到约 20 万到 40 万点的分块。`GT_DATABASE_MAX_POINTS=0` 表示不截断单个目标内的点。

如果仍然遇到 `SIGSEGV`，先降低目标计算规模，不要关闭 `gt_sampling`：

```bash
GT_DATABASE_TARGET_ELEMENTS=1000000 docker compose run --rm prepare
```

只有确认缓存不可信或想从零重建 ground-truth database 时，才执行：

```bash
docker compose run --rm reprepare
```

一键执行转换、准备和训练：

```bash
docker compose run --rm pipeline
```

## 预训练权重

这个 fork 不把第三方权重打进 Docker 镜像。权重属于实验输入，和 `labelCloud/`、`data/`、`output/` 一样通过宿主机目录挂载。默认挂载规则：

```text
宿主机 ./checkpoints  ->  容器 /workspace/checkpoints
```

准备目录：

```bash
mkdir -p checkpoints
```

官方公开权重入口：

```text
PV-RCNN KITTI:
https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-/view?usp=sharing

DSVT-Pillar nuScenes:
https://drive.google.com/file/d/10d7c-uJxg5w4GN-JmRBQi4gQDwHiOHxP/view?usp=drive_link

OpenPCDet model zoo:
https://github.com/open-mmlab/OpenPCDet#model-zoo

DSVT model zoo:
https://github.com/Haiyang-W/DSVT#main-results
```

说明：

- OpenPCDet 官方 model zoo 明确提供 KITTI LiDAR 模型下载，其中包括 `PV-RCNN`
- OpenPCDet 的 Waymo 表包含 `PV-RCNN++` 指标，但官方说明 Waymo 预训练权重因 Waymo Dataset License Agreement 不提供下载
- DSVT 官方仓库提供 nuScenes `DSVT(Pillar)` checkpoint；Waymo DSVT 只提供日志，不提供预训练权重
- 当前 pipeline 支持给 `pv_rcnn`、`pv_rcnn_plusplus`、`dsvt_pillar` 填任意兼容 `.pth`，但不保证第三方非官方权重的结构和命名匹配

使用 PV-RCNN KITTI 权重：

```text
MODEL=pv_rcnn
PRETRAINED_MODEL=/workspace/checkpoints/pv_rcnn_kitti.pth
EXTRA_TAG=pv_rcnn_kitti_ft
```

使用 DSVT-Pillar nuScenes 权重：

```text
MODEL=dsvt_pillar
PRETRAINED_MODEL=/workspace/checkpoints/dsvt_pillar_nuscenes.pth
EXTRA_TAG=dsvt_pillar_nuscenes_ft
```

使用自备 PV-RCNN++ 权重：

```text
MODEL=pv_rcnn_plusplus
PRETRAINED_MODEL=/workspace/checkpoints/pv_rcnn_plusplus_xxx.pth
EXTRA_TAG=pv_rcnnpp_pretrained_ft
```

`PRETRAINED_MODEL` 和 `CKPT` 的区别：

```text
PRETRAINED_MODEL  外部预训练初始化，只加载名称和 shape 匹配的模型参数，不加载 optimizer，适合微调
CKPT              本项目断点续训，严格加载模型和 optimizer，适合继续上次未完成训练
```

微调命令和普通训练相同：

```bash
docker compose run --rm pipeline
```

已完成 convert + prepare 后训练或继续训练：

```bash
docker compose run --rm train
```

`train` 不做转换和数据准备，只进入 OpenPCDet 训练流程。OpenPCDet 的训练脚本会在相同输出目录下查找已有 checkpoint 并继续训练；如果 `.env` 设置了 `CKPT`，则按该 checkpoint 恢复。如果要完全从头训练，使用 `retrain`、清理对应 `output/.../ckpt` 目录，或换 `EXTRA_TAG`。

已完成 convert + prepare 后清理当前实验训练输出并从头训练：

```bash
docker compose run --rm retrain
```

`retrain` 只清理当前 `MODEL + CFG_FILE + EXTRA_TAG` 对应的训练输出目录，不清理 `data/custom`、转换结果、prepare 缓存或其他模型输出。该 Compose 服务会传入空 `--ckpt` 覆盖 `.env` 的 `CKPT`，避免清理输出后又尝试恢复同一路径下已删除的 checkpoint。

训练完成后推理并写出预测 JSON：

```bash
docker compose run --rm infer
```

`infer` 默认使用最新的 `output/.../ckpt/checkpoint_epoch_*.pth`，默认只读取仓库根目录的 `./infer`，也就是容器内的 `/workspace/infer`。推理集不参与训练数据划分，由你自行准备 `.pcd` 文件。Docker 推理只写结构化预测结果，不直接裁切点云。输出到：

```text
output/predictions/labelcloud_pv_rcnn/default/
  summary.json
  geomap_20260601_130101/
    predictions.json
```

默认每个场景都会写 `predictions.json`，即使没有预测框，方便宿主机可视化脚本打开整帧点云。

对新点云推理时，放入：

```text
infer/
  geomap_20260601_130101.pcd
  geomap_20260601_130245.pcd
```

然后执行：

```bash
docker compose run --rm infer
```

把自动拆分出的 `val` 评估集复制到 `infer/` 做 `.pcd` 推理：

```bash
mkdir -p infer
find infer -maxdepth 1 -type f -name '*.pcd' -delete

while IFS= read -r id; do
  cp "labelCloud/pointClouds/${id}.pcd" "infer/${id}.pcd"
done < data/custom/ImageSets/val.txt
```

不要在日常命令里对 `infer/` 使用 `sudo`，否则目录或 `.pcd` 可能变成 root 所有，后续普通用户运行可视化、裁切或清理会遇到权限问题。如果之前已经用 sudo 创建过，先执行一次 `sudo chown -R "$USER:$USER" infer` 修正权限。

如果原始点云目录不在 `labelCloud/pointClouds`，指定实际目录：

```bash
SRC="/path/to/labelCloud/pointClouds"

mkdir -p infer
find infer -maxdepth 1 -type f -name '*.pcd' -delete

while IFS= read -r id; do
  cp "${SRC}/${id}.pcd" "infer/${id}.pcd"
done < data/custom/ImageSets/val.txt
```

也可以显式指定输入目录：

```bash
INPUT_DIR=/workspace/infer PRED_OUT_DIR=output/predictions/custom_run SCORE_THRESH=0.3 docker compose run --rm infer
```

如果需要指定某个 checkpoint：

```bash
CKPT=output/data/custom/cfgs/labelcloud_pv_rcnn/default/ckpt/checkpoint_epoch_80.pth docker compose run --rm infer
```

推理其他模型时使用同一个 `MODEL`：

```bash
MODEL=dsvt_pillar docker compose run --rm infer
```

## 宿主机可视化

宿主机可视化不集成到 Docker Compose。Docker 负责训练和推理，宿主机用 `uv` 管理 Open3D 等 GUI/查看依赖，避免 X11、WSL 和服务器显示环境影响容器。

可视化脚本只打开整帧点云并叠加 `predictions.json` 里的预测框，不负责裁切，也不打开裁切后的单体点云。

初始化宿主机工具环境：

```bash
uv sync
```

打开整帧点云并叠加 `predictions.json` 里的所有预测框：

```bash
uv run python tools/visualize_labelcloud_result.py output/predictions/labelcloud_pv_rcnn/default/geomap_20260601_130101
```

如果 `summary.json` 不在输出目录上层，或原始输入点云移动过，可以显式指定整帧点云：

```bash
uv run python tools/visualize_labelcloud_result.py output/predictions/labelcloud_pv_rcnn/default/geomap_20260601_130101 --input-file infer/geomap_20260601_130101.pcd
```

脚本会自动把容器里的 `/workspace/infer/<name>.pcd` 映射到宿主机当前仓库的 `infer/<name>.pcd`。如果推理输入不在默认 `infer/` 目录，使用 `--input-file` 指定原始整帧点云。

只显示分数不低于 `0.2` 的预测框：

```bash
uv run python tools/visualize_labelcloud_result.py output/predictions/labelcloud_pv_rcnn/default/geomap_20260601_130101 --score-thresh 0.2
```

## 宿主机裁切

裁切脚本按 `predictions.json` 从原始整帧点云中导出预测框内点。裁切是后处理，不放在 Docker 推理或可视化主流程里：

```bash
uv run python tools/crop_labelcloud_predictions.py output/predictions/labelcloud_pv_rcnn/default/geomap_20260601_130101
```

默认导出 `.pcd` 到场景目录。

指定裁切输出目录：

```bash
uv run python tools/crop_labelcloud_predictions.py output/predictions/labelcloud_pv_rcnn/default/geomap_20260601_130101 --output-dir output/crops/custom_run/geomap_20260601_130101
```

进入容器排查：

```bash
docker compose run --rm shell
```

## 训练输出

转换后的数据：

```text
data/custom/
  points/
  labels/
  ImageSets/
  cfgs/
  model_cache/
```

训练输出：

```text
output/data/custom/cfgs/labelcloud_pv_rcnn/default/
```

默认推理输出：

```text
output/predictions/labelcloud_pv_rcnn/default/
```

OpenPCDet 推理输出中最关键的是：

```text
pred_boxes:  N x 7, x y z dx dy dz heading
pred_scores: N
pred_labels: N
```

`pred_boxes` 会写入每个场景的 `predictions.json`，宿主机可视化工具可按需导出 3D box 内点云。

## 注意事项

- 不要运行容器时把整个仓库挂载到 `/workspace/OpenPCDet`，否则可能覆盖镜像构建阶段编译好的 CUDA 扩展
- `data/` 和 `output/` 是运行产物，不应打进镜像
- 当前自动化默认复用转换产物和 GT database；只有执行 `reconvert` 或 `reprepare` 才会重建相关产物
- 这个 fork 当前没有实现点级分割训练；宿主机裁切是按预测 3D box 保留框内点，不做点级边界分割
- 训练结束后的 eval 使用自动拆分出的 `val`，不参与参数更新；最终测试集应放在 `infer/` 中通过 `docker compose run --rm infer` 跑推理，再用宿主机工具可视化或裁切
- KITTI eval 会把自定义类别映射到 KITTI 的 `Car` 参与评估，多类别货物的训练标签仍保留原始类别；该评估结果只作为粗略参考。
