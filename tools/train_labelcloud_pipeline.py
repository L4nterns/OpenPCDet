import argparse
import os
import socket
import subprocess
import sys
import shutil
from pathlib import Path


MODEL_CHOICES = ('pv_rcnn', 'pv_rcnn_plusplus', 'dsvt_pillar', 'dsvt_voxel')
AUG_MODE_DISABLE_LISTS = {
    'full': ['placeholder'],
    'safe': ['gt_sampling', 'random_world_flip', 'random_world_rotation'],
    'none': ['gt_sampling', 'random_world_flip', 'random_world_rotation', 'random_world_scaling'],
}


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
        description='Convert labelCloud data and train OpenPCDet PV-RCNN in one command.'
    )
    parser.add_argument('--class-file', default=None, help='Path to labelCloud _classes.json.')
    parser.add_argument('--pointcloud-dir', default=None, help='Path to labelCloud pointClouds directory.')
    parser.add_argument('--label-dir', default=None, help='Path to labelCloud labels directory.')
    parser.add_argument('--out-dir', default='data/custom')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', default='auto', help='Total batch size. Use "auto" for 2 samples per GPU.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--num-gpus', default='auto', help='Use "auto" to train with all visible GPUs.')
    parser.add_argument('--master-port', type=int, default=None, help='Optional torch distributed master port.')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--aug-mode', choices=tuple(AUG_MODE_DISABLE_LISTS), default=os.environ.get('AUG_MODE', 'full'))
    parser.add_argument('--model', choices=MODEL_CHOICES, default='pv_rcnn')
    parser.add_argument('--extra-tag', default=os.environ.get('EXTRA_TAG'))
    parser.add_argument('--ckpt', default=os.environ.get('CKPT'), help='Optional checkpoint to resume from.')
    parser.add_argument(
        '--pretrained-model',
        default=os.environ.get('PRETRAINED_MODEL'),
        help='Optional pretrained weights to initialize matching model parameters.',
    )
    parser.add_argument('--cfg-file', default=os.environ.get('CFG_FILE'))
    parser.add_argument('--dataset-cfg', default=os.environ.get('DATASET_CFG'))
    parser.add_argument('--model-cfg', default=os.environ.get('MODEL_CFG'))
    parser.add_argument('--skip-convert', action='store_true')
    parser.add_argument('--skip-info', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--reset-output', action='store_true', help='Remove the target training output directory before training.')
    parser.add_argument('--gt-database-target-elements', type=int, default=4000000)
    parser.add_argument('--gt-database-min-chunk-size', type=int, default=50000)
    parser.add_argument('--gt-database-max-chunk-size', type=int, default=500000)
    parser.add_argument('--gt-database-max-points', type=int, default=0)
    parser.add_argument('--sample-points', type=int, default=int(os.environ.get('SAMPLE_POINTS', '-1')))
    parser.add_argument('--dsvt-iou-head', type=str_to_bool, default=str_to_bool(os.environ.get('DSVT_IOU_HEAD', 'true')))
    parser.add_argument(
        '--dsvt-iou-reg-loss',
        type=str_to_bool,
        default=str_to_bool(os.environ.get('DSVT_IOU_REG_LOSS', 'true')),
    )
    parser.add_argument(
        '--dsvt-use-iou-to-rectify-score',
        type=str_to_bool,
        default=str_to_bool(os.environ.get('DSVT_USE_IOU_TO_RECTIFY_SCORE', 'true')),
    )
    parser.add_argument('--rebuild-gt-database', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def run(cmd, cwd, dry_run=False):
    print('+ ' + ' '.join(str(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, check=True)


def default_extra_tag(model):
    return 'default'


def none_if_blank(value):
    return None if value == '' else value


def default_paths_for_out_dir(out_dir, model):
    out_path = Path(out_dir)
    dataset_cfg = (out_path / 'cfgs' / f'labelcloud_dataset_{model}.yaml').as_posix()
    model_cfg = (out_path / 'cfgs' / f'labelcloud_{model}.yaml').as_posix()
    prepare_dir = (out_path / 'model_cache' / model).as_posix()
    if out_path.is_absolute():
        cfg_file = model_cfg
    else:
        cfg_file = ('..' / out_path / 'cfgs' / f'labelcloud_{model}.yaml').as_posix()
    return dataset_cfg, model_cfg, cfg_file, prepare_dir


def set_cfgs_for_aug_mode(aug_mode):
    if aug_mode is None:
        return []
    disable_list = AUG_MODE_DISABLE_LISTS[aug_mode]
    return [
        'DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST',
        repr(disable_list),
    ]


def visible_gpu_count():
    try:
        import torch
    except ImportError:
        return 0
    return torch.cuda.device_count()


def resolve_num_gpus(value):
    if value == 'auto':
        return visible_gpu_count()
    num_gpus = int(value)
    if num_gpus < 0:
        raise ValueError('--num-gpus must be "auto" or a non-negative integer.')
    return num_gpus


def resolve_batch_size(value, num_gpus):
    if value == 'auto':
        return max(1, num_gpus) * 2
    batch_size = int(value)
    if batch_size <= 0:
        raise ValueError('--batch-size must be "auto" or a positive integer.')
    if num_gpus > 1 and batch_size % num_gpus != 0:
        raise ValueError('--batch-size must be divisible by the resolved GPU count.')
    return batch_size


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def train_output_dir(root, cfg_file, extra_tag):
    cfg_parts = Path(cfg_file).as_posix().split('/')
    exp_group = Path(*cfg_parts[1:-1]) if len(cfg_parts) > 2 else Path()
    return root / 'output' / exp_group / Path(cfg_file).stem / extra_tag


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.extra_tag = none_if_blank(args.extra_tag)
    args.cfg_file = none_if_blank(args.cfg_file)
    args.dataset_cfg = none_if_blank(args.dataset_cfg)
    args.model_cfg = none_if_blank(args.model_cfg)
    args.ckpt = none_if_blank(args.ckpt)
    args.pretrained_model = none_if_blank(args.pretrained_model)
    args.aug_mode = none_if_blank(args.aug_mode)
    extra_tag = args.extra_tag or default_extra_tag(args.model)
    default_dataset_cfg, default_model_cfg, default_cfg_file, default_prepare_dir = default_paths_for_out_dir(
        args.out_dir, args.model
    )
    dataset_cfg = args.dataset_cfg or default_dataset_cfg
    model_cfg = args.model_cfg or default_model_cfg
    cfg_file = args.cfg_file or default_cfg_file
    num_gpus = resolve_num_gpus(args.num_gpus)
    batch_size = resolve_batch_size(args.batch_size, num_gpus)

    if not args.skip_convert:
        missing = [
            name for name, value in [
                ('--class-file', args.class_file),
                ('--pointcloud-dir', args.pointcloud_dir),
                ('--label-dir', args.label_dir),
            ] if not value
        ]
        if missing:
            raise ValueError('Missing required arguments for conversion: ' + ', '.join(missing))

        convert_cmd = [
            sys.executable, 'tools/convert_labelcloud_to_custom.py',
            '--class-file', args.class_file,
            '--pointcloud-dir', args.pointcloud_dir,
            '--label-dir', args.label_dir,
            '--out_dir', args.out_dir,
            '--train-ratio', str(args.train_ratio),
            '--val-ratio', str(args.val_ratio),
            '--seed', str(args.seed),
            '--model', args.model,
            '--dataset-cfg', dataset_cfg,
            '--model-cfg', model_cfg,
            '--sample-points', str(args.sample_points),
            '--dsvt-iou-head', str(args.dsvt_iou_head),
            '--dsvt-iou-reg-loss', str(args.dsvt_iou_reg_loss),
            '--dsvt-use-iou-to-rectify-score', str(args.dsvt_use_iou_to_rectify_score),
        ]
        if args.overwrite:
            convert_cmd.append('--overwrite')
        run(convert_cmd, cwd=root, dry_run=args.dry_run)

    if not args.skip_info:
        info_cmd = [
            sys.executable, '-m', 'pcdet.datasets.custom.custom_dataset',
            'create_custom_infos', dataset_cfg,
            '--data-path', args.out_dir,
            '--save-path', default_prepare_dir,
            '--workers', str(args.workers),
            '--gt-database-target-elements', str(args.gt_database_target_elements),
            '--gt-database-min-chunk-size', str(args.gt_database_min_chunk_size),
            '--gt-database-max-chunk-size', str(args.gt_database_max_chunk_size),
            '--gt-database-max-points', str(args.gt_database_max_points),
        ]
        if args.rebuild_gt_database:
            info_cmd.append('--rebuild-gt-database')
        run(info_cmd, cwd=root, dry_run=args.dry_run)

    if args.skip_train:
        return

    if args.reset_output:
        output_dir = train_output_dir(root, cfg_file, extra_tag)
        print('+ rm -rf ' + str(output_dir))
        if output_dir.exists() and not args.dry_run:
            shutil.rmtree(output_dir)

    train_args = [
        '--cfg_file', cfg_file,
        '--batch_size', str(batch_size),
        '--workers', str(args.workers),
        '--extra_tag', extra_tag,
    ]
    if args.epochs is not None:
        train_args.extend(['--epochs', str(args.epochs)])
    if args.ckpt is not None:
        train_args.extend(['--ckpt', args.ckpt])
    if args.pretrained_model is not None:
        train_args.extend(['--pretrained_model', args.pretrained_model])
    set_cfgs = set_cfgs_for_aug_mode(args.aug_mode)
    if set_cfgs:
        train_args.extend(['--set', *set_cfgs])

    if num_gpus > 1:
        master_port = args.master_port if args.master_port is not None else find_free_port()
        train_cmd = [
            sys.executable, '-m', 'torch.distributed.run',
            '--nproc_per_node', str(num_gpus),
            '--master_port', str(master_port),
            'train.py',
            '--launcher', 'pytorch',
            *train_args,
        ]
    else:
        train_cmd = [sys.executable, 'train.py', *train_args]

    run(train_cmd, cwd=root / 'tools', dry_run=args.dry_run)


if __name__ == '__main__':
    main()
