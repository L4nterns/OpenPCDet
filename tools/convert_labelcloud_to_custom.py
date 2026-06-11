import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np


MODEL_CHOICES = ('pv_rcnn', 'pv_rcnn_plusplus', 'dsvt_pillar', 'dsvt_voxel')


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if value in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'Expected boolean value, got {value!r}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a labelCloud project to OpenPCDet custom dataset format.'
    )
    parser.add_argument('--labelcloud_root', default=None, help='Optional path to labelCloud project root.')
    parser.add_argument('--out_dir', default='data/custom', help='Output OpenPCDet custom dataset directory.')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class-file', default=None, help='Path to labelCloud _classes.json.')
    parser.add_argument('--pointcloud-dir', default=None, help='Path to labelCloud pointClouds directory.')
    parser.add_argument('--label-dir', default=None, help='Path to labelCloud labels directory.')
    parser.add_argument('--dataset-cfg', default=None)
    parser.add_argument('--model-cfg', default=None)
    parser.add_argument('--model', choices=MODEL_CHOICES, default='pv_rcnn')
    parser.add_argument('--voxel-xy', type=float, default=0.1)
    parser.add_argument('--range-margin', type=float, default=5.0)
    parser.add_argument('--min-points-filter', type=int, default=5)
    parser.add_argument('--sample-group-size', type=int, default=20)
    parser.add_argument(
        '--sample-points',
        type=int,
        default=-1,
        help='Limit points per frame in generated configs. Use -1 to keep all points.',
    )
    parser.add_argument(
        '--dsvt-iou-head',
        type=str_to_bool,
        default=True,
        help='Generate DSVT CenterHead iou branch. Disabling it also disables dependent DSVT IoU options.',
    )
    parser.add_argument(
        '--dsvt-iou-reg-loss',
        type=str_to_bool,
        default=True,
        help='Enable DSVT CenterHead IOU_REG_LOSS.',
    )
    parser.add_argument(
        '--dsvt-use-iou-to-rectify-score',
        type=str_to_bool,
        default=True,
        help='Enable DSVT post-processing score rectification with predicted IoU.',
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def resolve_inputs(args):
    if args.labelcloud_root:
        root = Path(args.labelcloud_root)
        class_file = Path(args.class_file) if args.class_file else root / '_classes.json'
        pointcloud_dir = Path(args.pointcloud_dir) if args.pointcloud_dir else root / 'pointClouds'
        label_dir = Path(args.label_dir) if args.label_dir else root / 'labels'
    else:
        if not (args.class_file and args.pointcloud_dir and args.label_dir):
            raise ValueError(
                'Provide either --labelcloud_root or all of --class-file, --pointcloud-dir, --label-dir.'
            )
        class_file = Path(args.class_file)
        pointcloud_dir = Path(args.pointcloud_dir)
        label_dir = Path(args.label_dir)

    for path, desc in [(class_file, 'class file'), (pointcloud_dir, 'pointcloud dir'), (label_dir, 'label dir')]:
        if not path.exists():
            raise FileNotFoundError(f'Missing {desc}: {path}')
    if not class_file.is_file():
        raise FileNotFoundError(f'Class file is not a file: {class_file}')
    if not pointcloud_dir.is_dir():
        raise NotADirectoryError(f'Pointcloud dir is not a directory: {pointcloud_dir}')
    if not label_dir.is_dir():
        raise NotADirectoryError(f'Label dir is not a directory: {label_dir}')

    return class_file, pointcloud_dir, label_dir


def load_classes(class_file):
    data = json.loads(class_file.read_text())
    classes = [item['name'] for item in data['classes']]
    if not classes:
        raise ValueError(f'No classes found in {class_file}')
    if len(set(classes)) != len(classes):
        raise ValueError(f'Duplicate class names found in {class_file}: {classes}')
    return classes


def normalize_heading_degrees(angle_deg):
    angle = math.radians(float(angle_deg))
    return (angle + math.pi) % (2 * math.pi) - math.pi


def scene_id_from_pcd_name(name):
    return Path(name).stem


def read_label_file(label_path, class_names):
    data = json.loads(label_path.read_text())
    rows = []
    extents = []
    dims_by_class = {name: [] for name in class_names}
    bottoms_by_class = {name: [] for name in class_names}

    for obj in data.get('objects', []):
        name = obj['name']
        if name not in class_names:
            raise ValueError(f'Unknown class {name!r} in {label_path}')

        c = obj['centroid']
        d = obj['dimensions']
        r = obj['rotations']
        if abs(float(r.get('x', 0.0))) > 1e-4 or abs(float(r.get('y', 0.0))) > 1e-4:
            raise ValueError(
                f'Only z-axis rotation is supported by OpenPCDet boxes; got non-zero x/y rotation in {label_path}'
            )
        x, y, z = float(c['x']), float(c['y']), float(c['z'])
        dx, dy, dz = float(d['length']), float(d['width']), float(d['height'])
        heading = normalize_heading_degrees(float(r['z']))

        rows.append((x, y, z, dx, dy, dz, heading, name))
        dims_by_class[name].append((dx, dy, dz))
        bottoms_by_class[name].append(z - dz / 2.0)

        radius = math.sqrt(dx * dx + dy * dy) / 2.0
        extents.append((x - radius, y - radius, z - dz / 2.0))
        extents.append((x + radius, y + radius, z + dz / 2.0))

    return data, rows, extents, dims_by_class, bottoms_by_class


def read_pcd_as_xyzrgb(path):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            'open3d is required to read PCD files. Use docker/Dockerfile.local-cu116, '
            'or install open3d in the current Python environment.'
        ) from exc

    pcd = o3d.io.read_point_cloud(str(path), remove_nan_points=True, remove_infinite_points=True)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'Failed to read XYZ points from {path}')

    colors = np.asarray(pcd.colors, dtype=np.float32)
    if colors.shape == xyz.shape and colors.size > 0:
        rgb = np.clip(colors, 0.0, 1.0).astype(np.float32)
    else:
        rgb = np.zeros_like(xyz, dtype=np.float32)

    return np.concatenate([xyz, rgb], axis=1).astype(np.float32, copy=False)


def split_scenes(scene_ids, train_ratio, val_ratio, seed):
    total = train_ratio + val_ratio
    if total <= 0:
        raise ValueError('At least one split ratio must be positive.')
    train_ratio, val_ratio = [x / total for x in (train_ratio, val_ratio)]

    scene_ids = list(scene_ids)
    random.Random(seed).shuffle(scene_ids)
    n = len(scene_ids)
    n_val = int(round(n * val_ratio))
    n_train = n - n_val

    if train_ratio > 0 and n_train == 0 and n > 0:
        n_train, n_val = 1, max(0, n_val - 1)
    if val_ratio > 0 and n_val == 0 and n > 1:
        n_train, n_val = n - 1, 1

    train = scene_ids[:n_train]
    val = scene_ids[n_train:n_train + n_val]
    return {'train': train, 'val': val}


def remove_path(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def clean_generated_outputs(out_dir, model=None):
    if model is not None:
        remove_path(model_cache_dir(out_dir, model))
        remove_path(Path(default_dataset_cfg_path(out_dir, model)))
        remove_path(Path(default_model_cfg_path(out_dir, model)))
        return

    clean_all_generated_outputs(out_dir)


def clean_all_generated_outputs(out_dir):
    for name in [
        'points', 'labels', 'ImageSets', 'cfgs',
        'custom_infos_train.pkl', 'custom_infos_val.pkl',
        'custom_dbinfos_train.pkl', 'gt_database',
        'conversion_summary.json',
    ]:
        remove_path(out_dir / name)


def ensure_output_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def output_is_stale(output_path, source_paths):
    if not output_path.exists():
        return True
    output_mtime = output_path.stat().st_mtime_ns
    return any(source.stat().st_mtime_ns > output_mtime for source in source_paths)


def write_text_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def save_npy_atomic(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'wb') as f:
        np.save(f, array)
    os.replace(tmp_path, path)


def npy_num_points(path, expected_num_features=6):
    if not path.exists():
        return None
    try:
        arr = np.load(path, mmap_mode='r')
    except (OSError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != expected_num_features:
        return None
    return int(arr.shape[0])


def fmt_list(values):
    return '[' + ', '.join(f'{float(v):.6g}' for v in values) + ']'


def yaml_list_str(values):
    return '[' + ', '.join(repr(v) for v in values) + ']'


def class_stats(class_names, dims_by_class, bottoms_by_class):
    stats = {}
    for name in class_names:
        dims = np.asarray(dims_by_class[name], dtype=np.float32)
        bottoms = np.asarray(bottoms_by_class[name], dtype=np.float32)
        if dims.size == 0:
            stats[name] = {
                'anchor_sizes': [[1.0, 1.0, 1.0]],
                'anchor_bottom_heights': [0.0],
            }
            continue

        qs = [50] if len(dims) < 10 else [25, 50, 75]
        anchor_sizes = np.percentile(dims, qs, axis=0)
        anchor_sizes = np.maximum(anchor_sizes, 0.05)
        bottom_qs = [50] if len(bottoms) < 10 else [25, 50, 75]
        bottom_heights = np.percentile(bottoms, bottom_qs).tolist()
        stats[name] = {
            'anchor_sizes': anchor_sizes.tolist(),
            'anchor_bottom_heights': bottom_heights,
        }
    return stats


def aligned_point_cloud_range(extents, margin, voxel_xy):
    if not extents:
        return [-80.0, -80.0, -5.0, 80.0, 80.0, 7.0], [voxel_xy, voxel_xy, 0.3]

    arr = np.asarray(extents, dtype=np.float32)
    mins = arr.min(axis=0) - margin
    maxs = arr.max(axis=0) + margin

    def align_axis(lo, hi, voxel):
        lo = math.floor(float(lo) / voxel) * voxel
        hi = math.ceil(float(hi) / voxel) * voxel
        cells = int(round((hi - lo) / voxel))
        rem = cells % 16
        if rem:
            hi += (16 - rem) * voxel
        return lo, hi

    x_min, x_max = align_axis(mins[0], maxs[0], voxel_xy)
    y_min, y_max = align_axis(mins[1], maxs[1], voxel_xy)

    z_min = math.floor(float(mins[2]) * 10.0) / 10.0
    z_max = math.ceil(float(maxs[2]) * 10.0) / 10.0
    z_span = max(z_max - z_min, 4.0)
    voxel_z = round(z_span / 40.0, 4)
    z_max = z_min + voxel_z * 40

    return [x_min, y_min, z_min, x_max, y_max, z_max], [voxel_xy, voxel_xy, voxel_z]


def data_path_for_train(out_dir):
    out_path = Path(out_dir)
    if out_path.is_absolute():
        return out_path.as_posix()
    return ('..' / out_path).as_posix()


def model_cache_dir(out_dir, model):
    return Path(out_dir) / 'model_cache' / model


def model_cache_path_for_train(out_dir, model):
    out_path = Path(out_dir)
    cache_path = out_path / 'model_cache' / model
    if out_path.is_absolute():
        return cache_path.as_posix()
    return (Path('model_cache') / model).as_posix()


def default_dataset_cfg_path(out_dir, model='pv_rcnn'):
    return (Path(out_dir) / 'cfgs' / f'labelcloud_dataset_{model}.yaml').as_posix()


def model_cfg_name(model):
    return f'labelcloud_{model}.yaml'


def default_model_cfg_path(out_dir, model):
    return (Path(out_dir) / 'cfgs' / model_cfg_name(model)).as_posix()


def write_dataset_cfg(
    path, class_names, pc_range, voxel_size, min_points, sample_group_size, out_dir,
    model='pv_rcnn', sample_points=-1,
):
    filters = [f'{name}:{min_points}' for name in class_names]
    sample_groups = [f'{name}:{sample_group_size}' for name in class_names]
    map_items = ',\n    '.join(f'{name!r}: "Car"' for name in class_names)
    sample_points_processor = sample_points_processor_text(sample_points, indent=4)

    text = f"""CLASS_NAMES: {yaml_list_str(class_names)}
DATASET: 'CustomDataset'
DATA_PATH: {data_path_for_train(out_dir)!r}

POINT_CLOUD_RANGE: {fmt_list(pc_range)}

MAP_CLASS_TO_KITTI: {{
    {map_items}
}}

DATA_SPLIT: {{
    'train': train,
    'test': val
}}

INFO_PATH: {{
    'train': [{model_cache_path_for_train(out_dir, model)}/custom_infos_train.pkl],
    'test': [{model_cache_path_for_train(out_dir, model)}/custom_infos_val.pkl],
}}

POINT_FEATURE_ENCODING: {{
    encoding_type: absolute_coordinates_encoding,
    used_feature_list: ['x', 'y', 'z', 'r', 'g', 'b'],
    src_feature_list: ['x', 'y', 'z', 'r', 'g', 'b'],
}}

DATA_AUGMENTOR:
    DISABLE_AUG_LIST: ['placeholder']
    AUG_CONFIG_LIST:
        - NAME: gt_sampling
          USE_ROAD_PLANE: False
          DB_INFO_PATH:
              - {model_cache_path_for_train(out_dir, model)}/custom_dbinfos_train.pkl
          PREPARE: {{
             filter_by_min_points: {yaml_list_str(filters)},
          }}

          SAMPLE_GROUPS: {yaml_list_str(sample_groups)}
          NUM_POINT_FEATURES: 6
          DATABASE_WITH_FAKELIDAR: False
          REMOVE_EXTRA_WIDTH: [0.0, 0.0, 0.0]
          LIMIT_WHOLE_SCENE: True

        - NAME: random_world_flip
          ALONG_AXIS_LIST: ['x', 'y']

        - NAME: random_world_rotation
          WORLD_ROT_ANGLE: [-0.78539816, 0.78539816]

        - NAME: random_world_scaling
          WORLD_SCALE_RANGE: [0.95, 1.05]

DATA_PROCESSOR:
    - NAME: mask_points_and_boxes_outside_range
      REMOVE_OUTSIDE_BOXES: True

    - NAME: shuffle_points
      SHUFFLE_ENABLED: {{
        'train': True,
        'test': False
      }}
{sample_points_processor}

    - NAME: transform_points_to_voxels
      VOXEL_SIZE: {fmt_list(voxel_size)}
      MAX_POINTS_PER_VOXEL: 5
      MAX_NUMBER_OF_VOXELS: {{
        'train': 150000,
        'test': 150000
      }}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def sample_points_processor_text(sample_points, indent):
    if sample_points is None or sample_points < 0:
        return ''
    spaces = ' ' * indent
    child_spaces = ' ' * (indent + 2)
    return f"""
{spaces}- NAME: sample_points
{child_spaces}NUM_POINTS: {{
{child_spaces}  'train': {sample_points},
{child_spaces}  'test': {sample_points}
{child_spaces}}}
"""


def anchor_config_text(class_names, stats):
    anchor_blocks = []
    for name in class_names:
        cls_stats = stats[name]
        anchor_sizes = '[' + ', '.join(fmt_list(x) for x in cls_stats['anchor_sizes']) + ']'
        bottom_heights = fmt_list(cls_stats['anchor_bottom_heights'])
        anchor_blocks.append(f"""            {{
                'class_name': '{name}',
                'anchor_sizes': {anchor_sizes},
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': {bottom_heights},
                'align_center': False,
                'feature_map_stride': 8,
                'matched_threshold': 0.5,
                'unmatched_threshold': 0.35
            }}""")
    return ',\n'.join(anchor_blocks)


def base_config_path_for_model(model_cfg_path, dataset_cfg_path):
    model_cfg_path = Path(model_cfg_path)
    path = Path(dataset_cfg_path)
    if path.is_absolute():
        return path.as_posix()
    try:
        return os.path.relpath(path, start=model_cfg_path.parent).replace(os.sep, '/')
    except ValueError:
        return path.as_posix()


def write_pv_rcnn_model_cfg(path, dataset_cfg_path, class_names, stats):
    anchor_config = anchor_config_text(class_names, stats)
    rel_dataset = base_config_path_for_model(path, dataset_cfg_path)

    text = f"""CLASS_NAMES: {yaml_list_str(class_names)}

DATA_CONFIG:
    _BASE_CONFIG_: {rel_dataset}

MODEL:
    NAME: PVRCNN

    VFE:
        NAME: MeanVFE

    BACKBONE_3D:
        NAME: VoxelBackBone8x

    MAP_TO_BEV:
        NAME: HeightCompression
        NUM_BEV_FEATURES: 256

    BACKBONE_2D:
        NAME: BaseBEVBackbone

        LAYER_NUMS: [5, 5]
        LAYER_STRIDES: [1, 2]
        NUM_FILTERS: [128, 256]
        UPSAMPLE_STRIDES: [1, 2]
        NUM_UPSAMPLE_FILTERS: [256, 256]

    DENSE_HEAD:
        NAME: AnchorHeadSingle
        CLASS_AGNOSTIC: False

        USE_DIRECTION_CLASSIFIER: True
        DIR_OFFSET: 0.78539
        DIR_LIMIT_OFFSET: 0.0
        NUM_DIR_BINS: 2

        ANCHOR_GENERATOR_CONFIG: [
{anchor_config}
        ]

        TARGET_ASSIGNER_CONFIG:
            NAME: AxisAlignedTargetAssigner
            POS_FRACTION: -1.0
            SAMPLE_SIZE: 512
            NORM_BY_NUM_EXAMPLES: False
            MATCH_HEIGHT: False
            BOX_CODER: ResidualCoder

        LOSS_CONFIG:
            LOSS_WEIGHTS: {{
                'cls_weight': 1.0,
                'loc_weight': 2.0,
                'dir_weight': 0.2,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }}

    PFE:
        NAME: VoxelSetAbstraction
        POINT_SOURCE: raw_points
        NUM_KEYPOINTS: 4096
        NUM_OUTPUT_FEATURES: 128
        SAMPLE_METHOD: FPS

        FEATURES_SOURCE: ['bev', 'x_conv3', 'x_conv4', 'raw_points']
        SA_LAYER:
            raw_points:
                MLPS: [[16, 16], [16, 16]]
                POOL_RADIUS: [0.4, 0.8]
                NSAMPLE: [16, 16]
            x_conv1:
                DOWNSAMPLE_FACTOR: 1
                MLPS: [[16, 16], [16, 16]]
                POOL_RADIUS: [0.4, 0.8]
                NSAMPLE: [16, 16]
            x_conv2:
                DOWNSAMPLE_FACTOR: 2
                MLPS: [[32, 32], [32, 32]]
                POOL_RADIUS: [0.8, 1.2]
                NSAMPLE: [16, 32]
            x_conv3:
                DOWNSAMPLE_FACTOR: 4
                MLPS: [[64, 64], [64, 64]]
                POOL_RADIUS: [1.2, 2.4]
                NSAMPLE: [16, 32]
            x_conv4:
                DOWNSAMPLE_FACTOR: 8
                MLPS: [[64, 64], [64, 64]]
                POOL_RADIUS: [2.4, 4.8]
                NSAMPLE: [16, 32]

    POINT_HEAD:
        NAME: PointHeadSimple
        CLS_FC: [256, 256]
        CLASS_AGNOSTIC: True
        USE_POINT_FEATURES_BEFORE_FUSION: True
        TARGET_CONFIG:
            GT_EXTRA_WIDTH: [0.2, 0.2, 0.2]
        LOSS_CONFIG:
            LOSS_REG: smooth-l1
            LOSS_WEIGHTS: {{
                'point_cls_weight': 1.0,
            }}

    ROI_HEAD:
        NAME: PVRCNNHead
        CLASS_AGNOSTIC: True

        SHARED_FC: [256, 256]
        CLS_FC: [256, 256]
        REG_FC: [256, 256]
        DP_RATIO: 0.3

        NMS_CONFIG:
            TRAIN:
                NMS_TYPE: nms_gpu
                MULTI_CLASSES_NMS: False
                NMS_PRE_MAXSIZE: 9000
                NMS_POST_MAXSIZE: 512
                NMS_THRESH: 0.8
            TEST:
                NMS_TYPE: nms_gpu
                MULTI_CLASSES_NMS: False
                NMS_PRE_MAXSIZE: 4096
                NMS_POST_MAXSIZE: 300
                NMS_THRESH: 0.85

        ROI_GRID_POOL:
            GRID_SIZE: 6
            MLPS: [[64, 64], [64, 64]]
            POOL_RADIUS: [0.8, 1.6]
            NSAMPLE: [16, 16]
            POOL_METHOD: max_pool

        TARGET_CONFIG:
            BOX_CODER: ResidualCoder
            ROI_PER_IMAGE: 128
            FG_RATIO: 0.5

            SAMPLE_ROI_BY_EACH_CLASS: True
            CLS_SCORE_TYPE: roi_iou

            CLS_FG_THRESH: 0.75
            CLS_BG_THRESH: 0.25
            CLS_BG_THRESH_LO: 0.1
            HARD_BG_RATIO: 0.8

            REG_FG_THRESH: 0.55

        LOSS_CONFIG:
            CLS_LOSS: BinaryCrossEntropy
            REG_LOSS: smooth-l1
            CORNER_LOSS_REGULARIZATION: True
            LOSS_WEIGHTS: {{
                'rcnn_cls_weight': 1.0,
                'rcnn_reg_weight': 1.0,
                'rcnn_corner_weight': 1.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }}

    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
        SCORE_THRESH: 0.1
        OUTPUT_RAW_SCORE: False

        EVAL_METRIC: kitti

        NMS_CONFIG:
            MULTI_CLASSES_NMS: False
            NMS_TYPE: nms_gpu
            NMS_THRESH: 0.1
            NMS_PRE_MAXSIZE: 4096
            NMS_POST_MAXSIZE: 500

OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 2
    NUM_EPOCHS: 80

    OPTIMIZER: adam_onecycle
    LR: 0.01
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9

    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001

    LR_WARMUP: False
    WARMUP_EPOCH: 1

    GRAD_NORM_CLIP: 10
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_pv_rcnn_plusplus_model_cfg(path, dataset_cfg_path, class_names, pc_range):
    rel_dataset = base_config_path_for_model(path, dataset_cfg_path)
    class_head = '[' + yaml_list_str(class_names) + ']'
    post_range = [pc_range[0], pc_range[1], pc_range[2] - 5.0, pc_range[3], pc_range[4], pc_range[5] + 5.0]

    text = f"""CLASS_NAMES: {yaml_list_str(class_names)}

DATA_CONFIG:
    _BASE_CONFIG_: {rel_dataset}

MODEL:
    NAME: PVRCNNPlusPlus

    VFE:
        NAME: MeanVFE

    BACKBONE_3D:
        NAME: VoxelBackBone8x

    MAP_TO_BEV:
        NAME: HeightCompression
        NUM_BEV_FEATURES: 256

    BACKBONE_2D:
        NAME: BaseBEVBackbone

        LAYER_NUMS: [5, 5]
        LAYER_STRIDES: [1, 2]
        NUM_FILTERS: [128, 256]
        UPSAMPLE_STRIDES: [1, 2]
        NUM_UPSAMPLE_FILTERS: [256, 256]

    DENSE_HEAD:
        NAME: CenterHead
        CLASS_AGNOSTIC: False
        CLASS_NAMES_EACH_HEAD: {class_head}
        SHARED_CONV_CHANNEL: 64
        USE_BIAS_BEFORE_NORM: True
        NUM_HM_CONV: 2
        SEPARATE_HEAD_CFG:
            HEAD_ORDER: ['center', 'center_z', 'dim', 'rot']
            HEAD_DICT: {{
                'center': {{'out_channels': 2, 'num_conv': 2}},
                'center_z': {{'out_channels': 1, 'num_conv': 2}},
                'dim': {{'out_channels': 3, 'num_conv': 2}},
                'rot': {{'out_channels': 2, 'num_conv': 2}},
            }}

        TARGET_ASSIGNER_CONFIG:
            FEATURE_MAP_STRIDE: 8
            NUM_MAX_OBJS: 500
            GAUSSIAN_OVERLAP: 0.1
            MIN_RADIUS: 2

        LOSS_CONFIG:
            LOSS_WEIGHTS: {{
                'cls_weight': 1.0,
                'loc_weight': 2.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }}

        POST_PROCESSING:
            SCORE_THRESH: 0.01
            POST_CENTER_LIMIT_RANGE: {fmt_list(post_range)}
            MAX_OBJ_PER_SAMPLE: 500
            NMS_CONFIG:
                NMS_TYPE: nms_gpu
                NMS_THRESH: 0.7
                NMS_PRE_MAXSIZE: 4096
                NMS_POST_MAXSIZE: 500

    PFE:
        NAME: VoxelSetAbstraction
        POINT_SOURCE: raw_points
        NUM_KEYPOINTS: 4096
        NUM_OUTPUT_FEATURES: 90
        SAMPLE_METHOD: SPC
        SPC_SAMPLING:
            NUM_SECTORS: 6
            SAMPLE_RADIUS_WITH_ROI: 1.6

        FEATURES_SOURCE: ['bev', 'x_conv3', 'x_conv4', 'raw_points']
        SA_LAYER:
            raw_points:
                NAME: VectorPoolAggregationModuleMSG
                NUM_GROUPS: 2
                LOCAL_AGGREGATION_TYPE: local_interpolation
                NUM_REDUCED_CHANNELS: 3
                NUM_CHANNELS_OF_LOCAL_AGGREGATION: 32
                MSG_POST_MLPS: [32]
                FILTER_NEIGHBOR_WITH_ROI: True
                RADIUS_OF_NEIGHBOR_WITH_ROI: 2.4

                GROUP_CFG_0:
                    NUM_LOCAL_VOXEL: [2, 2, 2]
                    MAX_NEIGHBOR_DISTANCE: 0.2
                    NEIGHBOR_NSAMPLE: -1
                    POST_MLPS: [32, 32]
                GROUP_CFG_1:
                    NUM_LOCAL_VOXEL: [3, 3, 3]
                    MAX_NEIGHBOR_DISTANCE: 0.4
                    NEIGHBOR_NSAMPLE: -1
                    POST_MLPS: [32, 32]

            x_conv3:
                DOWNSAMPLE_FACTOR: 4
                INPUT_CHANNELS: 64
                NAME: VectorPoolAggregationModuleMSG
                NUM_GROUPS: 2
                LOCAL_AGGREGATION_TYPE: local_interpolation
                NUM_REDUCED_CHANNELS: 32
                NUM_CHANNELS_OF_LOCAL_AGGREGATION: 32
                MSG_POST_MLPS: [128]
                FILTER_NEIGHBOR_WITH_ROI: True
                RADIUS_OF_NEIGHBOR_WITH_ROI: 4.0

                GROUP_CFG_0:
                    NUM_LOCAL_VOXEL: [3, 3, 3]
                    MAX_NEIGHBOR_DISTANCE: 1.2
                    NEIGHBOR_NSAMPLE: -1
                    POST_MLPS: [64, 64]
                GROUP_CFG_1:
                    NUM_LOCAL_VOXEL: [3, 3, 3]
                    MAX_NEIGHBOR_DISTANCE: 2.4
                    NEIGHBOR_NSAMPLE: -1
                    POST_MLPS: [64, 64]

            x_conv4:
                DOWNSAMPLE_FACTOR: 8
                INPUT_CHANNELS: 64
                NAME: VectorPoolAggregationModuleMSG
                NUM_GROUPS: 2
                LOCAL_AGGREGATION_TYPE: local_interpolation
                NUM_REDUCED_CHANNELS: 32
                NUM_CHANNELS_OF_LOCAL_AGGREGATION: 32
                MSG_POST_MLPS: [128]
                FILTER_NEIGHBOR_WITH_ROI: True
                RADIUS_OF_NEIGHBOR_WITH_ROI: 6.4

                GROUP_CFG_0:
                    NUM_LOCAL_VOXEL: [3, 3, 3]
                    MAX_NEIGHBOR_DISTANCE: 2.4
                    NEIGHBOR_NSAMPLE: -1
                    POST_MLPS: [64, 64]
                GROUP_CFG_1:
                    NUM_LOCAL_VOXEL: [3, 3, 3]
                    MAX_NEIGHBOR_DISTANCE: 4.8
                    NEIGHBOR_NSAMPLE: -1
                    POST_MLPS: [64, 64]

    POINT_HEAD:
        NAME: PointHeadSimple
        CLS_FC: [256, 256]
        CLASS_AGNOSTIC: True
        USE_POINT_FEATURES_BEFORE_FUSION: True
        TARGET_CONFIG:
            GT_EXTRA_WIDTH: [0.2, 0.2, 0.2]
        LOSS_CONFIG:
            LOSS_REG: smooth-l1
            LOSS_WEIGHTS: {{
                'point_cls_weight': 1.0,
            }}

    ROI_HEAD:
        NAME: PVRCNNHead
        CLASS_AGNOSTIC: True

        SHARED_FC: [256, 256]
        CLS_FC: [256, 256]
        REG_FC: [256, 256]
        DP_RATIO: 0.3

        NMS_CONFIG:
            TRAIN:
                NMS_TYPE: nms_gpu
                MULTI_CLASSES_NMS: False
                NMS_PRE_MAXSIZE: 9000
                NMS_POST_MAXSIZE: 512
                NMS_THRESH: 0.8
            TEST:
                NMS_TYPE: nms_gpu
                MULTI_CLASSES_NMS: False
                NMS_PRE_MAXSIZE: 1024
                NMS_POST_MAXSIZE: 100
                NMS_THRESH: 0.7
                SCORE_THRESH: 0.1

        ROI_GRID_POOL:
            GRID_SIZE: 6
            NAME: VectorPoolAggregationModuleMSG
            NUM_GROUPS: 2
            LOCAL_AGGREGATION_TYPE: voxel_random_choice
            NUM_REDUCED_CHANNELS: 30
            NUM_CHANNELS_OF_LOCAL_AGGREGATION: 32
            MSG_POST_MLPS: [128]

            GROUP_CFG_0:
                NUM_LOCAL_VOXEL: [3, 3, 3]
                MAX_NEIGHBOR_DISTANCE: 0.8
                NEIGHBOR_NSAMPLE: 32
                POST_MLPS: [64, 64]
            GROUP_CFG_1:
                NUM_LOCAL_VOXEL: [3, 3, 3]
                MAX_NEIGHBOR_DISTANCE: 1.6
                NEIGHBOR_NSAMPLE: 32
                POST_MLPS: [64, 64]

        TARGET_CONFIG:
            BOX_CODER: ResidualCoder
            ROI_PER_IMAGE: 128
            FG_RATIO: 0.5
            SAMPLE_ROI_BY_EACH_CLASS: True
            CLS_SCORE_TYPE: roi_iou
            CLS_FG_THRESH: 0.75
            CLS_BG_THRESH: 0.25
            CLS_BG_THRESH_LO: 0.1
            HARD_BG_RATIO: 0.8
            REG_FG_THRESH: 0.55

        LOSS_CONFIG:
            CLS_LOSS: BinaryCrossEntropy
            REG_LOSS: smooth-l1
            CORNER_LOSS_REGULARIZATION: True
            LOSS_WEIGHTS: {{
                'rcnn_cls_weight': 1.0,
                'rcnn_reg_weight': 1.0,
                'rcnn_corner_weight': 1.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }}

    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
        SCORE_THRESH: 0.1
        OUTPUT_RAW_SCORE: False
        EVAL_METRIC: kitti

        NMS_CONFIG:
            MULTI_CLASSES_NMS: False
            NMS_TYPE: nms_gpu
            NMS_THRESH: 0.7
            NMS_PRE_MAXSIZE: 4096
            NMS_POST_MAXSIZE: 500

OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 2
    NUM_EPOCHS: 100
    OPTIMIZER: adam_onecycle
    LR: 0.01
    WEIGHT_DECAY: 0.001
    MOMENTUM: 0.9
    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001
    LR_WARMUP: False
    WARMUP_EPOCH: 1
    GRAD_NORM_CLIP: 10
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def dsvt_shape(pc_range, voxel_size):
    x = int(round((pc_range[3] - pc_range[0]) / voxel_size[0]))
    y = int(round((pc_range[4] - pc_range[1]) / voxel_size[1]))
    z = int(round((pc_range[5] - pc_range[2]) / voxel_size[2]))
    return [x, y, z]


def dsvt_aligned_range(pc_range, voxel_size, pillar=False):
    x_cells = int(math.ceil((pc_range[3] - pc_range[0]) / voxel_size[0] / 12.0) * 12)
    y_cells = int(math.ceil((pc_range[4] - pc_range[1]) / voxel_size[1] / 12.0) * 12)
    z_cells = 1 if pillar else int(math.ceil((pc_range[5] - pc_range[2]) / voxel_size[2]))
    x_max = pc_range[0] + x_cells * voxel_size[0]
    y_max = pc_range[1] + y_cells * voxel_size[1]
    z_max = pc_range[2] + z_cells * voxel_size[2]
    return [pc_range[0], pc_range[1], pc_range[2], x_max, y_max, z_max]


def normalize_dsvt_iou_options(dsvt_iou_head, dsvt_iou_reg_loss, dsvt_use_iou_to_rectify_score):
    dsvt_iou_head = bool(dsvt_iou_head)
    if not dsvt_iou_head:
        return False, False, False
    return True, bool(dsvt_iou_reg_loss), bool(dsvt_use_iou_to_rectify_score)


def dsvt_voxel_z_layout(z_cells):
    stage0_z = max(1, int(z_cells))
    stage1_z = max(1, int(math.ceil(stage0_z / 4)))
    stage2_z = max(1, int(math.ceil(stage1_z / 4)))
    final_stride_z = stage2_z
    return {
        'downsample_stride': [[1, 1, 4], [1, 1, 4], [1, 1, final_stride_z]],
        'window_shape': [[12, 12, stage0_z], [12, 12, stage1_z], [12, 12, stage2_z], [12, 12, 1]],
    }


def write_dsvt_model_cfg(
    path, dataset_cfg_path, class_names, pc_range, model, sample_points=-1,
    dsvt_iou_head=True, dsvt_iou_reg_loss=True, dsvt_use_iou_to_rectify_score=True,
):
    rel_dataset = base_config_path_for_model(path, dataset_cfg_path)
    class_head = '[' + yaml_list_str(class_names) + ']'
    pillar = model == 'dsvt_pillar'
    voxel_size = [0.32, 0.32, max(0.1, pc_range[5] - pc_range[2])] if pillar else [0.32, 0.32, 0.1875]
    dsvt_range = dsvt_aligned_range(pc_range, voxel_size, pillar=pillar)
    shape = dsvt_shape(dsvt_range, voxel_size)
    post_range = [dsvt_range[0], dsvt_range[1], dsvt_range[2] - 8.0, dsvt_range[3], dsvt_range[4], dsvt_range[5] + 8.0]
    sample_points_processor = sample_points_processor_text(sample_points, indent=8)
    dsvt_iou_head, dsvt_iou_reg_loss, dsvt_use_iou_to_rectify_score = normalize_dsvt_iou_options(
        dsvt_iou_head, dsvt_iou_reg_loss, dsvt_use_iou_to_rectify_score,
    )
    iou_head_text = "                'iou': {'out_channels': 1, 'num_conv': 2},\n" if dsvt_iou_head else ''
    iou_reg_loss = str(dsvt_iou_reg_loss)
    use_iou_to_rectify_score = str(dsvt_use_iou_to_rectify_score)

    if pillar:
        map_to_bev_input_shape = shape
        input_layer = f"""            sparse_shape: {fmt_list(shape)}
            downsample_stride: []
            d_model: [192]
            set_info: [[36, 4]]
            window_shape: [[12, 12, 1]]
            hybrid_factor: [2, 2, 1]
            shifts_list: [[[0, 0, 0], [6, 6, 0]]]
            normalize_pos: False"""
        backbone_blocks = """        block_name: ['DSVTBlock']
        set_info: [[36, 4]]
        d_model: [192]
        nhead: [8]
        dim_feedforward: [384]
        dropout: 0.0
        activation: gelu
        output_shape: [%d, %d]
        conv_out_channel: 192
        USE_CHECKPOINT: True""" % (shape[0], shape[1])
    else:
        z_layout = dsvt_voxel_z_layout(shape[2])
        map_to_bev_input_shape = [shape[0], shape[1], 1]
        input_layer = f"""            sparse_shape: {fmt_list(shape)}
            downsample_stride: {z_layout['downsample_stride']}
            d_model: [192, 192, 192, 192]
            set_info: [[48, 1], [48, 1], [48, 1], [48, 1]]
            window_shape: {z_layout['window_shape']}
            hybrid_factor: [2, 2, 1]
            shifts_list: [[[0, 0, 0], [6, 6, 0]], [[0, 0, 0], [6, 6, 0]], [[0, 0, 0], [6, 6, 0]], [[0, 0, 0], [6, 6, 0]]]
            normalize_pos: False"""
        backbone_blocks = """        block_name: ['DSVTBlock','DSVTBlock','DSVTBlock','DSVTBlock']
        set_info: [[48, 1], [48, 1], [48, 1], [48, 1]]
        d_model: [192, 192, 192, 192]
        nhead: [8, 8, 8, 8]
        dim_feedforward: [384, 384, 384, 384]
        dropout: 0.0
        activation: gelu
        reduction_type: 'attention'
        output_shape: [%d, %d]
        conv_out_channel: 192
        USE_CHECKPOINT: True""" % (shape[0], shape[1])

    text = f"""CLASS_NAMES: {yaml_list_str(class_names)}

DATA_CONFIG:
    _BASE_CONFIG_: {rel_dataset}
    POINT_CLOUD_RANGE: {fmt_list(dsvt_range)}
    POINTS_TANH_DIM: [3, 4]

    DATA_PROCESSOR:
        - NAME: mask_points_and_boxes_outside_range
          REMOVE_OUTSIDE_BOXES: True

        - NAME: shuffle_points
          SHUFFLE_ENABLED: {{
            'train': True,
            'test': False
          }}
{sample_points_processor}

        - NAME: transform_points_to_voxels_placeholder
          VOXEL_SIZE: {fmt_list(voxel_size)}

MODEL:
    NAME: CenterPoint

    VFE:
        NAME: DynamicVoxelVFE
        WITH_DISTANCE: False
        USE_ABSLOTE_XYZ: True
        USE_NORM: True
        NUM_FILTERS: [192, 192]

    BACKBONE_3D:
        NAME: DSVT
        INPUT_LAYER:
{input_layer}

{backbone_blocks}

    MAP_TO_BEV:
        NAME: PointPillarScatter3d
        INPUT_SHAPE: {fmt_list(map_to_bev_input_shape)}
        NUM_BEV_FEATURES: 192

    BACKBONE_2D:
        NAME: BaseBEVResBackbone
        LAYER_NUMS: [1, 2, 2]
        LAYER_STRIDES: [1, 2, 2]
        NUM_FILTERS: [128, 128, 256]
        UPSAMPLE_STRIDES: [1, 2, 4]
        NUM_UPSAMPLE_FILTERS: [128, 128, 128]

    DENSE_HEAD:
        NAME: CenterHead
        CLASS_AGNOSTIC: False
        CLASS_NAMES_EACH_HEAD: {class_head}
        SHARED_CONV_CHANNEL: 64
        USE_BIAS_BEFORE_NORM: False
        NUM_HM_CONV: 2
        BN_EPS: 0.001
        BN_MOM: 0.01
        SEPARATE_HEAD_CFG:
            HEAD_ORDER: ['center', 'center_z', 'dim', 'rot']
            HEAD_DICT: {{
                'center': {{'out_channels': 2, 'num_conv': 2}},
                'center_z': {{'out_channels': 1, 'num_conv': 2}},
                'dim': {{'out_channels': 3, 'num_conv': 2}},
                'rot': {{'out_channels': 2, 'num_conv': 2}},
{iou_head_text}            }}

        TARGET_ASSIGNER_CONFIG:
            FEATURE_MAP_STRIDE: 1
            NUM_MAX_OBJS: 500
            GAUSSIAN_OVERLAP: 0.1
            MIN_RADIUS: 2

        IOU_REG_LOSS: {iou_reg_loss}

        LOSS_CONFIG:
            LOSS_WEIGHTS: {{
                'cls_weight': 1.0,
                'loc_weight': 2.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }}

        POST_PROCESSING:
            SCORE_THRESH: 0.1
            POST_CENTER_LIMIT_RANGE: {fmt_list(post_range)}
            MAX_OBJ_PER_SAMPLE: 500
            USE_IOU_TO_RECTIFY_SCORE: {use_iou_to_rectify_score}
            IOU_RECTIFIER: {[0.68 for _ in class_names]}

            NMS_CONFIG:
                NMS_TYPE: class_specific_nms
                NMS_THRESH: {[0.7 for _ in class_names]}
                NMS_PRE_MAXSIZE: {[4096 for _ in class_names]}
                NMS_POST_MAXSIZE: {[500 for _ in class_names]}

    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
        EVAL_METRIC: kitti

OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 2
    NUM_EPOCHS: 80
    OPTIMIZER: adam_onecycle
    LR: 0.003
    WEIGHT_DECAY: 0.05
    MOMENTUM: 0.9
    MOMS: [0.95, 0.85]
    PCT_START: 0.1
    DIV_FACTOR: 100
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001
    LR_WARMUP: False
    WARMUP_EPOCH: 1
    GRAD_NORM_CLIP: 10
    LOSS_SCALE_FP16: 32.0
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_model_cfg(
    path, dataset_cfg_path, class_names, stats, pc_range, model, sample_points=-1,
    dsvt_iou_head=True, dsvt_iou_reg_loss=True, dsvt_use_iou_to_rectify_score=True,
):
    if model == 'pv_rcnn':
        write_pv_rcnn_model_cfg(path, dataset_cfg_path, class_names, stats)
    elif model == 'pv_rcnn_plusplus':
        write_pv_rcnn_plusplus_model_cfg(path, dataset_cfg_path, class_names, pc_range)
    elif model in ('dsvt_pillar', 'dsvt_voxel'):
        write_dsvt_model_cfg(
            path, dataset_cfg_path, class_names, pc_range, model, sample_points=sample_points,
            dsvt_iou_head=dsvt_iou_head,
            dsvt_iou_reg_loss=dsvt_iou_reg_loss,
            dsvt_use_iou_to_rectify_score=dsvt_use_iou_to_rectify_score,
        )
    else:
        raise ValueError(f'Unsupported model: {model}')


def write_default_model_cfgs(
    out_dir, class_names, stats, pc_range, voxel_size, min_points, sample_group_size, sample_points=-1,
    dsvt_iou_head=True, dsvt_iou_reg_loss=True, dsvt_use_iou_to_rectify_score=True,
):
    dataset_cfgs = {}
    model_cfgs = {}
    for model in MODEL_CHOICES:
        dataset_cfg = default_dataset_cfg_path(out_dir, model)
        model_cfg = default_model_cfg_path(out_dir, model)
        write_dataset_cfg(
            Path(dataset_cfg), class_names, pc_range, voxel_size,
            min_points=min_points,
            sample_group_size=sample_group_size,
            out_dir=out_dir,
            model=model,
            sample_points=sample_points,
        )
        write_model_cfg(
            Path(model_cfg), dataset_cfg, class_names, stats, pc_range, model, sample_points=sample_points,
            dsvt_iou_head=dsvt_iou_head,
            dsvt_iou_reg_loss=dsvt_iou_reg_loss,
            dsvt_use_iou_to_rectify_score=dsvt_use_iou_to_rectify_score,
        )
        dataset_cfgs[model] = dataset_cfg
        model_cfgs[model] = model_cfg
    return dataset_cfgs, model_cfgs


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    default_dataset_cfg = default_dataset_cfg_path(args.out_dir, args.model)
    default_model_cfg = default_model_cfg_path(args.out_dir, args.model)
    dataset_cfg = args.dataset_cfg or default_dataset_cfg
    model_cfg = args.model_cfg or default_model_cfg
    class_file, pointcloud_dir, label_dir = resolve_inputs(args)
    class_names = load_classes(class_file)

    if args.overwrite:
        clean_generated_outputs(out_dir, model=args.model)

    points_out = out_dir / 'points'
    labels_out = out_dir / 'labels'
    imagesets_out = out_dir / 'ImageSets'
    generated_dirs = [points_out, labels_out, imagesets_out]
    for p in generated_dirs:
        ensure_output_dir(p)

    label_files = sorted(label_dir.glob('*.json'))
    label_files = [p for p in label_files if p.name != '_classes.json']
    if not label_files:
        raise FileNotFoundError(f'No label JSON files found in {label_dir}')

    label_records = []
    seen_scene_ids = {}
    for label_path in label_files:
        data, rows, extents, dims_part, bottoms_part = read_label_file(label_path, class_names)
        pcd_name = data.get('filename') or (label_path.stem + '.pcd')
        pcd_path = pointcloud_dir / pcd_name
        if not pcd_path.exists():
            raise FileNotFoundError(f'Missing PCD for {label_path}: {pcd_path}')

        scene_id = scene_id_from_pcd_name(pcd_name)
        if scene_id in seen_scene_ids:
            raise ValueError(
                f'Duplicate scene id {scene_id!r}: {pcd_name!r} from '
                f'{label_path} conflicts with {seen_scene_ids[scene_id]}. '
                'Rename the PCD file or provide unique filenames.'
            )
        seen_scene_ids[scene_id] = label_path
        label_records.append((label_path, rows, extents, dims_part, bottoms_part, pcd_path, scene_id))

    all_extents = []
    dims_by_class = {name: [] for name in class_names}
    bottoms_by_class = {name: [] for name in class_names}
    scene_ids = []

    print(f'Converting {len(label_records)} scenes with classes: {class_names}')
    for _label_path, rows, extents, dims_part, bottoms_part, pcd_path, scene_id in label_records:
        scene_ids.append(scene_id)

        points_path = points_out / f'{scene_id}.npy'
        points_count = npy_num_points(points_path)
        converted_points = False
        if points_count is None or output_is_stale(points_path, [pcd_path]):
            points = read_pcd_as_xyzrgb(pcd_path)
            save_npy_atomic(points_path, points)
            points_count = points.shape[0]
            converted_points = True

        label_path_out = labels_out / f'{scene_id}.txt'
        label_lines = []
        for row in rows:
            x, y, z, dx, dy, dz, heading, name = row
            label_lines.append(
                f'{x:.8f} {y:.8f} {z:.8f} {dx:.8f} {dy:.8f} {dz:.8f} {heading:.8f} {name}\n'
            )
        converted_labels = False
        if output_is_stale(label_path_out, [label_path, class_file]):
            write_text_atomic(label_path_out, ''.join(label_lines))
            converted_labels = True

        all_extents.extend(extents)
        for name in class_names:
            dims_by_class[name].extend(dims_part[name])
            bottoms_by_class[name].extend(bottoms_part[name])
        status = []
        if converted_points:
            status.append('points')
        if converted_labels:
            status.append('labels')
        status = '+'.join(status) if status else 'up-to-date'
        print(f'  {scene_id}: points={points_count} boxes={len(rows)} {status}')

    splits = split_scenes(scene_ids, args.train_ratio, args.val_ratio, args.seed)
    for split_name, ids in splits.items():
        (imagesets_out / f'{split_name}.txt').write_text(''.join(f'{x}\n' for x in ids))

    pc_range, voxel_size = aligned_point_cloud_range(all_extents, args.range_margin, args.voxel_xy)
    stats = class_stats(class_names, dims_by_class, bottoms_by_class)

    if args.overwrite:
        write_dataset_cfg(
            Path(dataset_cfg), class_names, pc_range, voxel_size,
            min_points=args.min_points_filter,
            sample_group_size=args.sample_group_size,
            out_dir=args.out_dir,
            model=args.model,
            sample_points=args.sample_points,
        )
        write_model_cfg(
            Path(model_cfg), dataset_cfg, class_names, stats, pc_range, args.model,
            sample_points=args.sample_points,
            dsvt_iou_head=args.dsvt_iou_head,
            dsvt_iou_reg_loss=args.dsvt_iou_reg_loss,
            dsvt_use_iou_to_rectify_score=args.dsvt_use_iou_to_rectify_score,
        )
        dataset_cfgs = {args.model: dataset_cfg}
        model_cfgs = {args.model: model_cfg}
    else:
        dataset_cfgs, model_cfgs = write_default_model_cfgs(
            args.out_dir, class_names, stats, pc_range, voxel_size,
            min_points=args.min_points_filter,
            sample_group_size=args.sample_group_size,
            sample_points=args.sample_points,
            dsvt_iou_head=args.dsvt_iou_head,
            dsvt_iou_reg_loss=args.dsvt_iou_reg_loss,
            dsvt_use_iou_to_rectify_score=args.dsvt_use_iou_to_rectify_score,
        )

    if not args.overwrite and (
        Path(dataset_cfg).as_posix() != Path(default_dataset_cfg).as_posix()
        or Path(model_cfg).as_posix() != Path(default_model_cfg).as_posix()
    ):
        write_dataset_cfg(
            Path(dataset_cfg), class_names, pc_range, voxel_size,
            min_points=args.min_points_filter,
            sample_group_size=args.sample_group_size,
            out_dir=args.out_dir,
            model=args.model,
            sample_points=args.sample_points,
        )
        write_model_cfg(
            Path(model_cfg), dataset_cfg, class_names, stats, pc_range, args.model,
            sample_points=args.sample_points,
            dsvt_iou_head=args.dsvt_iou_head,
            dsvt_iou_reg_loss=args.dsvt_iou_reg_loss,
            dsvt_use_iou_to_rectify_score=args.dsvt_use_iou_to_rectify_score,
        )

    summary = {
        'classes': class_names,
        'num_scenes': len(scene_ids),
        'splits': {k: len(v) for k, v in splits.items()},
        'num_boxes_by_class': {k: len(v) for k, v in dims_by_class.items()},
        'point_cloud_range': pc_range,
        'voxel_size': voxel_size,
        'dataset_cfg': dataset_cfg,
        'dataset_cfgs': dataset_cfgs,
        'model_cfg': model_cfg,
        'model_cfgs': model_cfgs,
        'model': args.model,
    }
    write_text_atomic(out_dir / 'conversion_summary.json', json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
