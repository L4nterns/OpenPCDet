import _init_path
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


MODEL_CHOICES = ('pv_rcnn', 'pv_rcnn_plusplus', 'dsvt_pillar', 'dsvt_voxel')


class LabelCloudInferenceDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, root_path, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names,
            training=False, root_path=root_path, logger=logger
        )
        root_path = Path(root_path)
        if root_path.is_dir():
            self.sample_file_list = sorted(
                [p for p in root_path.iterdir() if p.suffix.lower() == '.pcd']
            )
        else:
            self.sample_file_list = [root_path]

        if not self.sample_file_list:
            raise FileNotFoundError(f'No .pcd point clouds found in {root_path}')

    def __len__(self):
        return len(self.sample_file_list)

    def prepare_points(self, points, frame_id):
        return self.prepare_data({
            'points': points,
            'frame_id': frame_id,
        })

    def __getitem__(self, index):
        sample_file = self.sample_file_list[index]
        return self.prepare_points(load_points(sample_file), sample_file.stem)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run OpenPCDet inference for labelCloud custom data and write prediction JSON.'
    )
    parser.add_argument('--model', choices=MODEL_CHOICES, default='pv_rcnn')
    parser.add_argument('--cfg-file', default=os.environ.get('CFG_FILE'))
    parser.add_argument('--ckpt', default=os.environ.get('CKPT', 'auto'), help='Checkpoint path, or "auto" for the latest training checkpoint.')
    parser.add_argument('--input-dir', default='infer', help='Directory or single .pcd point cloud.')
    parser.add_argument('--output-dir', default=os.environ.get('PRED_OUT_DIR'))
    parser.add_argument('--score-thresh', type=float, default=0.3)
    parser.add_argument('--extra-tag', default=os.environ.get('EXTRA_TAG'))
    return parser.parse_args()


def default_cfg_file(model):
    return f'data/custom/cfgs/labelcloud_{model}.yaml'


def default_extra_tag(model):
    return 'default'


def default_output_dir(cfg_file, extra_tag):
    return (Path('output') / 'predictions' / Path(cfg_file).stem / extra_tag).as_posix()


def none_if_blank(value):
    return None if value == '' else value


def resolve_input_path(value):
    return Path(value)


def load_pcd_as_xyzrgb(path):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError('open3d is required to read PCD files.') from exc

    pcd = o3d.io.read_point_cloud(str(path), remove_nan_points=True, remove_infinite_points=True)
    xyz = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    if colors.shape == xyz.shape and colors.size > 0:
        rgb = np.clip(colors, 0.0, 1.0).astype(np.float32)
    else:
        rgb = np.zeros_like(xyz, dtype=np.float32)
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32, copy=False)


def load_points(path):
    suffix = path.suffix.lower()
    if suffix == '.pcd':
        return load_pcd_as_xyzrgb(path)
    raise ValueError(f'Unsupported point cloud extension: {path}')


def checkpoint_epoch(path):
    match = re.search(r'checkpoint_epoch_(\d+)\.pth$', path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(ckpt_dir):
    candidates = sorted(ckpt_dir.glob('checkpoint_epoch_*.pth'), key=lambda p: (checkpoint_epoch(p), p.stat().st_mtime_ns))
    if candidates:
        return candidates[-1]

    latest = ckpt_dir / 'latest_model.pth'
    if latest.exists():
        return latest

    all_ckpts = sorted(ckpt_dir.glob('*.pth'), key=lambda p: p.stat().st_mtime_ns)
    if all_ckpts:
        return all_ckpts[-1]
    raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')


def auto_checkpoint_path(cfg_file, extra_tag):
    cfg_path = Path(cfg_file)
    if not cfg_path.exists():
        raise FileNotFoundError(f'Missing cfg file: {cfg_file}')

    cfg_tag = cfg_path.stem
    raw_parts = [part for part in cfg_path.as_posix().split('/') if part and part != '.']
    normalized_parts = list(raw_parts)
    while normalized_parts and normalized_parts[0] == '..':
        normalized_parts = normalized_parts[1:]

    candidates = []
    if len(normalized_parts) > 1:
        candidates.append(Path('output') / Path(*normalized_parts[:-1]) / cfg_tag / extra_tag / 'ckpt')
    if len(raw_parts) > 2:
        train_py_group = raw_parts[1:-1]
        candidates.append(Path('output') / Path(*train_py_group) / cfg_tag / extra_tag / 'ckpt')

    seen = set()
    errors = []
    for ckpt_dir in candidates:
        ckpt_dir = Path(ckpt_dir)
        if ckpt_dir in seen:
            continue
        seen.add(ckpt_dir)
        try:
            return latest_checkpoint(ckpt_dir)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    raise FileNotFoundError('No checkpoint found. Tried:\n' + '\n'.join(errors))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)


def prediction_payload(frame_id, boxes, scores, labels, class_names):
    objects = []
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        label_idx = int(label) - 1
        name = class_names[label_idx] if 0 <= label_idx < len(class_names) else str(label)
        objects.append({
            'index': idx,
            'name': name,
            'label': int(label),
            'score': float(score),
            'box': {
                'x': float(box[0]),
                'y': float(box[1]),
                'z': float(box[2]),
                'dx': float(box[3]),
                'dy': float(box[4]),
                'dz': float(box[5]),
                'heading': float(box[6]),
            },
        })
    return {'frame_id': frame_id, 'objects': objects}


def main():
    args = parse_args()
    cfg_file = none_if_blank(args.cfg_file) or default_cfg_file(args.model)
    ckpt = none_if_blank(args.ckpt) or 'auto'
    extra_tag = none_if_blank(args.extra_tag) or default_extra_tag(args.model)
    cfg_from_yaml_file(cfg_file, cfg)
    logger = common_utils.create_logger()
    input_path = resolve_input_path(args.input_dir)
    output_dir = Path(none_if_blank(args.output_dir) or default_output_dir(cfg_file, extra_tag))

    logger.info(f'model={args.model}')
    logger.info(f'cfg_file={cfg_file}')
    logger.info(f'ckpt={ckpt}')
    logger.info(f'input={input_path}')
    logger.info(f'output={output_dir}')

    ckpt_path = auto_checkpoint_path(cfg_file, extra_tag) if ckpt == 'auto' else Path(ckpt)
    logger.info(f'resolved_ckpt={ckpt_path}')

    dataset = LabelCloudInferenceDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=input_path,
        logger=logger,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    summary = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample_file = dataset.sample_file_list[idx]
            frame_id = sample_file.stem
            original_points = load_points(sample_file)

            data_dict = dataset.collate_batch([dataset.prepare_points(original_points, frame_id)])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_boxes = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
            pred_scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
            pred_labels = pred_dicts[0]['pred_labels'].detach().cpu().numpy()
            keep = pred_scores >= args.score_thresh
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            scene_dir = output_dir / frame_id
            objects = prediction_payload(frame_id, pred_boxes, pred_scores, pred_labels, cfg.CLASS_NAMES)

            write_json(scene_dir / 'predictions.json', objects)

            logger.info(f'{idx + 1}/{len(dataset)} {frame_id}: boxes={len(objects["objects"])}')
            summary.append({
                'frame_id': frame_id,
                'input_file': sample_file.as_posix(),
                'num_predictions': len(objects['objects']),
                'scene_dir': scene_dir.as_posix(),
            })

    write_json(output_dir / 'summary.json', {
        'model': args.model,
        'cfg_file': cfg_file,
        'ckpt': ckpt_path.as_posix(),
        'input': input_path.as_posix(),
        'score_thresh': args.score_thresh,
        'scenes': summary,
    })


if __name__ == '__main__':
    main()
