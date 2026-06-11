import copy
import hashlib
import pickle
import os
import shutil
from pathlib import Path

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate


class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]

        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

        self.custom_infos = []
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

    def include_data(self, mode):
        self.logger.info('Loading Custom dataset.')
        custom_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)

        self.custom_infos.extend(custom_infos)
        self.logger.info('Total samples for CUSTOM dataset: %d' % (len(custom_infos)))

    def get_label(self, idx):
        label_file = self.root_path / 'labels' / ('%s.txt' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = [line for line in f.readlines() if line.strip()]

        if len(lines) == 0:
            return np.zeros((0, 7), dtype=np.float32), np.array([])

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(line_list[:-1])
            gt_names.append(line_list[-1])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_lidar(self, idx):
        lidar_file = self.root_path / 'points' / ('%s.npy' % idx)
        assert lidar_file.exists()
        point_features = np.load(lidar_file)
        return point_features

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.custom_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)

        info = copy.deepcopy(self.custom_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            'frame_id': sample_idx,
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.custom_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    @staticmethod
    def get_db_file_num_points(filepath, num_features):
        if not filepath.exists():
            return None
        bytes_per_point = num_features * np.dtype(np.float32).itemsize
        file_size = filepath.stat().st_size
        if bytes_per_point <= 0 or file_size % bytes_per_point != 0:
            return None
        return file_size // bytes_per_point

    @staticmethod
    def resolve_gt_database_chunk_size(num_boxes, target_elements, min_chunk_size, max_chunk_size):
        if min_chunk_size <= 0 or max_chunk_size <= 0:
            raise ValueError('GT database chunk sizes must be positive.')
        if min_chunk_size > max_chunk_size:
            raise ValueError('GT database min chunk size cannot be larger than max chunk size.')
        if target_elements <= 0:
            return int(max_chunk_size)
        if num_boxes <= 0:
            return int(max_chunk_size)
        chunk_size = max(1, int(target_elements) // max(1, int(num_boxes)))
        return max(int(min_chunk_size), min(int(max_chunk_size), chunk_size))

    def create_groundtruth_database(
        self, info_path=None, used_classes=None, split='train',
        target_elements=4000000, min_chunk_size=50000, max_chunk_size=500000,
        max_points_per_object=0, resume=True, database_save_path=None, db_info_save_path=None
    ):
        import torch

        database_save_path = Path(database_save_path) if database_save_path else (
            Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        )
        db_info_save_path = Path(db_info_save_path) if db_info_save_path else (
            Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)
        )

        if not resume:
            if database_save_path.exists():
                shutil.rmtree(database_save_path)
            if db_info_save_path.exists():
                db_info_save_path.unlink()
        database_save_path.mkdir(parents=True, exist_ok=True)
        db_classes = used_classes if used_classes is not None else self.class_names
        all_db_infos = {name: [] for name in db_classes}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            lidar_path = self.root_path / 'points' / ('%s.npy' % sample_idx)
            label_path = self.root_path / 'labels' / ('%s.txt' % sample_idx)
            source_mtime = max(lidar_path.stat().st_mtime_ns, label_path.stat().st_mtime_ns)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            points = None
            num_features = info['point_cloud'].get('num_features')
            if num_features is None:
                points = self.get_lidar(sample_idx)
                num_features = points.shape[1]
            missing_indices = []
            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                existing_num_points = self.get_db_file_num_points(filepath, num_features)
                if existing_num_points is not None and filepath.stat().st_mtime_ns >= source_mtime:
                    if (used_classes is None) or names[i] in used_classes:
                        db_path = str(filepath.relative_to(self.root_path))
                        db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                                   'box3d_lidar': gt_boxes[i], 'num_points_in_gt': existing_num_points}
                        all_db_infos[names[i]].append(db_info)
                    continue
                missing_indices.append(i)

            if not missing_indices:
                continue

            if points is None:
                points = self.get_lidar(sample_idx)
            missing_boxes = gt_boxes[missing_indices]
            gt_points_list = [[] for _ in missing_indices]
            chunk_size = self.resolve_gt_database_chunk_size(
                len(missing_indices), target_elements, min_chunk_size, max_chunk_size
            )
            print('  points=%d boxes=%d missing=%d chunk_size=%d' % (
                points.shape[0], num_obj, len(missing_indices), chunk_size
            ))

            for start in range(0, points.shape[0], chunk_size):
                end = min(start + chunk_size, points.shape[0])
                points_chunk = points[start:end]
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points_chunk[:, 0:3]), torch.from_numpy(missing_boxes)
                ).numpy()
                for local_i in range(len(missing_indices)):
                    obj_points = points_chunk[point_indices[local_i] > 0]
                    if obj_points.shape[0] > 0:
                        gt_points_list[local_i].append(obj_points)

            for local_i, i in enumerate(missing_indices):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                if gt_points_list[local_i]:
                    gt_points = np.concatenate(gt_points_list[local_i], axis=0)
                else:
                    gt_points = np.zeros((0, points.shape[1]), dtype=points.dtype)

                if max_points_per_object > 0 and gt_points.shape[0] > max_points_per_object:
                    seed_bytes = hashlib.blake2b(f'{sample_idx}:{i}'.encode('utf-8'), digest_size=8).digest()
                    seed = int.from_bytes(seed_bytes, byteorder='little') % (2 ** 32)
                    rng = np.random.default_rng(seed)
                    choice = rng.choice(gt_points.shape[0], max_points_per_object, replace=False)
                    gt_points = gt_points[choice]

                gt_points[:, :3] -= gt_boxes[i, :3]
                tmp_filepath = filepath.with_suffix(filepath.suffix + '.tmp')
                with open(tmp_filepath, 'wb') as f:
                    gt_points.tofile(f)
                os.replace(tmp_filepath, filepath)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    all_db_infos[names[i]].append(db_info)

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        tmp_db_info_save_path = db_info_save_path.with_suffix(db_info_save_path.suffix + '.tmp')
        with open(tmp_db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
        os.replace(tmp_db_info_save_path, db_info_save_path)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)


def create_custom_infos(
    dataset_cfg, class_names, data_path, save_path, workers=4,
    target_elements=4000000, min_chunk_size=50000, max_chunk_size=500000,
    max_points_per_object=0, resume_gt_database=True
):
    data_path = Path(data_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    dataset = CustomDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('custom_infos_%s.pkl' % train_split)
    val_filename = save_path / ('custom_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    custom_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(custom_infos_train, f)
    print('Custom info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    custom_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(custom_infos_val, f)
    print('Custom info train file is saved to %s' % val_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        train_filename, split=train_split,
        target_elements=target_elements,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        max_points_per_object=max_points_per_object,
        resume=resume_gt_database,
        database_save_path=save_path / 'gt_database',
        db_info_save_path=save_path / ('custom_dbinfos_%s.pkl' % train_split),
    )
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':
        import argparse
        import yaml
        from easydict import EasyDict

        parser = argparse.ArgumentParser(description='Create OpenPCDet custom dataset infos.')
        parser.add_argument('command', choices=['create_custom_infos'])
        parser.add_argument('dataset_cfg')
        parser.add_argument('--data-path', default='data/custom')
        parser.add_argument('--save-path', default=None)
        parser.add_argument('--workers', type=int, default=4)
        parser.add_argument('--gt-database-target-elements', type=int, default=4000000)
        parser.add_argument('--gt-database-min-chunk-size', type=int, default=50000)
        parser.add_argument('--gt-database-max-chunk-size', type=int, default=500000)
        parser.add_argument('--gt-database-max-points', type=int, default=0)
        parser.add_argument('--rebuild-gt-database', action='store_true')
        args = parser.parse_args()

        dataset_cfg = EasyDict(yaml.safe_load(open(args.dataset_cfg)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        class_names = dataset_cfg.get('CLASS_NAMES', ['Vehicle', 'Pedestrian', 'Cyclist'])
        data_path = Path(args.data_path)
        save_path = Path(args.save_path) if args.save_path else data_path
        if not data_path.is_absolute():
            data_path = ROOT_DIR / data_path
        if not save_path.is_absolute():
            save_path = ROOT_DIR / save_path
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            data_path=data_path,
            save_path=save_path,
            workers=args.workers,
            target_elements=args.gt_database_target_elements,
            min_chunk_size=args.gt_database_min_chunk_size,
            max_chunk_size=args.gt_database_max_chunk_size,
            max_points_per_object=args.gt_database_max_points,
            resume_gt_database=not args.rebuild_gt_database,
        )
