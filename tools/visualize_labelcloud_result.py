#!/usr/bin/env python3
import argparse
import configparser
import json
import shutil
import tempfile
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np
import open3d as o3d


DEFAULT_LABELCLOUD_CONFIGS = [
    Path.home() / 'Downloads' / 'labelCloud_weifang' / 'config.ini',
    Path.home() / 'downloads' / 'labelCloud_weifang' / 'config.ini',
    Path('/mnt/c/Users/linqi/Downloads/labelCloud_weifang/config.ini'),
    Path('C:/Users/linqi/Downloads/labelCloud_weifang/config.ini'),
]

BOX_COLORS = [
    [0.0, 1.0, 0.0],
    [0.0, 0.75, 1.0],
    [1.0, 0.85, 0.0],
    [1.0, 0.35, 0.35],
]


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize labelCloud inference scenes with predicted boxes.')
    parser.add_argument('path', help='A scene prediction directory or predictions.json.')
    parser.add_argument('--input-file', default=None, help='Original full-scene .pcd. Needed only when it cannot be resolved from summary.json.')
    parser.add_argument('--score-thresh', type=float, default=None, help='Optional score filter for displayed boxes.')
    parser.add_argument(
        '--labelcloud-config',
        default=find_default_labelcloud_config(),
        help='labelCloud config.ini used for Open3D display defaults.')
    parser.add_argument('--point-size', type=float, default=None)
    parser.add_argument('--background-color', default=None)
    parser.add_argument('--colorless-color', default=None)
    parser.add_argument(
        '--keep-perspective',
        action='store_true',
        help='Use perspective projection instead of labelCloud-style flat view.')
    parser.add_argument(
        '--view',
        choices=('top', 'free'),
        default='top',
        help='Initial camera view. "top" opens an XY top-down view.')
    return parser.parse_args()


def find_default_labelcloud_config():
    for path in DEFAULT_LABELCLOUD_CONFIGS:
        if path.exists():
            return path.as_posix()
    return None


def parse_rgb(value, default):
    if value is None:
        return np.array(default, dtype=np.float64)
    parts = [x.strip() for x in str(value).split(',')]
    if len(parts) != 3:
        raise ValueError(f'Expected RGB triplet, got: {value!r}')
    return np.array([float(x) for x in parts], dtype=np.float64)


def load_display_options(args):
    point_size = 0.5
    background_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    colorless_color = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    keep_perspective = False

    cfg_path = Path(args.labelcloud_config) if args.labelcloud_config else None
    if cfg_path and cfg_path.exists():
        config = configparser.ConfigParser()
        config.read(cfg_path, encoding='utf-8')
        point_size = config.getfloat('POINTCLOUD', 'point_size',
                                     fallback=point_size)
        background_color = parse_rgb(
            config.get('USER_INTERFACE', 'background_color', fallback=None),
            background_color)
        colorless_color = parse_rgb(
            config.get('POINTCLOUD', 'colorless_color', fallback=None),
            colorless_color)
        keep_perspective = config.getboolean(
            'USER_INTERFACE', 'keep_perspective', fallback=keep_perspective)

    if args.point_size is not None:
        point_size = args.point_size
    if args.background_color is not None:
        background_color = parse_rgb(args.background_color, background_color)
    if args.colorless_color is not None:
        colorless_color = parse_rgb(args.colorless_color, colorless_color)
    if args.keep_perspective:
        keep_perspective = True

    return {
        'point_size': point_size,
        'background_color': background_color,
        'colorless_color': colorless_color,
        'keep_perspective': keep_perspective,
        'view': args.view,
    }


def geometry_bounds(geometries):
    mins = []
    maxs = []
    for geometry in geometries:
        bbox = geometry.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        if np.all(np.isfinite(min_bound)) and np.all(np.isfinite(max_bound)):
            mins.append(min_bound)
            maxs.append(max_bound)
    if not mins:
        return np.zeros(3), np.ones(3)
    return np.min(np.asarray(mins), axis=0), np.max(np.asarray(maxs), axis=0)


def apply_view_options(vis, display, geometries):
    view_control = vis.get_view_control()
    if display['keep_perspective']:
        pass
    elif hasattr(view_control, 'change_field_of_view'):
        view_control.change_field_of_view(step=-90.0)

    if display['view'] == 'top':
        min_bound, max_bound = geometry_bounds(geometries)
        view_control.set_lookat(((min_bound + max_bound) / 2.0).tolist())
        view_control.set_front([0.0, 0.0, 1.0])
        view_control.set_up([0.0, 1.0, 0.0])
        view_control.set_zoom(0.7)


def open3d_read_point_cloud(path):
    path = Path(path)
    try:
        str(path).encode('ascii')
        read_path = path
        temp_dir = None
    except UnicodeEncodeError:
        temp_dir = tempfile.TemporaryDirectory()
        read_path = Path(temp_dir.name) / f'input{path.suffix}'
        shutil.copy2(path, read_path)

    try:
        return o3d.io.read_point_cloud(
            str(read_path), remove_nan_points=True, remove_infinite_points=True)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def load_points(path):
    path = Path(path)
    if path.suffix.lower() == '.pcd':
        pcd = open3d_read_point_cloud(path)
        xyz = np.asarray(pcd.points, dtype=np.float64)
        colors = np.asarray(pcd.colors, dtype=np.float64)
        if colors.shape == xyz.shape and colors.size > 0:
            return xyz, np.clip(colors, 0.0, 1.0)
        return xyz, None

    raise ValueError(f'Unsupported point cloud extension: {path}')


def make_point_cloud(path, colorless_color):
    xyz, colors = load_points(path)
    if xyz.size == 0:
        raise ValueError(f'Point cloud has no points: {path}')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color(colorless_color.tolist())
    return pcd


def find_predictions_path(path):
    path = Path(path)
    if path.is_file() and path.name == 'predictions.json':
        return path
    if path.is_dir() and (path / 'predictions.json').exists():
        return path / 'predictions.json'
    return None


def load_predictions(predictions_path, score_thresh=None):
    payload = json.loads(predictions_path.read_text())
    objects = payload.get('objects', [])
    if score_thresh is not None:
        objects = [obj for obj in objects if float(obj.get('score', 0.0)) >= score_thresh]
    return payload, objects


def find_summary_path(scene_dir):
    for parent in [scene_dir, *scene_dir.parents]:
        candidate = parent / 'summary.json'
        if candidate.exists():
            return candidate
    return None


def host_input_candidates(input_file):
    input_file = Path(input_file)
    candidates = [input_file]

    normalized = str(input_file).replace('\\', '/')
    if normalized.startswith('/workspace/'):
        workspace_path = PurePosixPath(normalized)
        rel_parts = workspace_path.parts[2:]
        if rel_parts:
            if rel_parts[0] == 'OpenPCDet':
                candidates.append(Path.cwd() / Path(*rel_parts[1:]))
            else:
                candidates.append(Path.cwd() / Path(*rel_parts))

    if not input_file.is_absolute():
        candidates.append(Path.cwd() / input_file)

    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def resolve_existing_input_file(input_file):
    for candidate in host_input_candidates(input_file):
        if candidate.exists():
            return candidate
    tried = '\n'.join(str(candidate) for candidate in host_input_candidates(input_file))
    raise FileNotFoundError(f'Input file does not exist. Tried:\n{tried}')


def default_input_candidates(frame_id):
    yield Path('infer') / f'{frame_id}.pcd'


def resolve_input_file(scene_dir, frame_id, explicit_input_file):
    if explicit_input_file:
        return resolve_existing_input_file(explicit_input_file)

    summary_path = find_summary_path(scene_dir)
    if summary_path is not None:
        summary = json.loads(summary_path.read_text())
        for scene in summary.get('scenes', []):
            if scene.get('frame_id') == frame_id:
                try:
                    return resolve_existing_input_file(scene['input_file'])
                except FileNotFoundError:
                    break

    for candidate in default_input_candidates(frame_id):
        if candidate.exists():
            return candidate
    raise FileNotFoundError('Could not resolve the original scene point cloud. Pass --input-file explicitly.')


def box_from_object(obj):
    box = obj['box']
    return np.array([
        box['x'], box['y'], box['z'],
        box['dx'], box['dy'], box['dz'],
        box['heading'],
    ], dtype=np.float64)


def make_box_geometry(obj, index):
    box = box_from_object(obj)
    center = box[:3]
    extent = box[3:6]
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, box[6]])
    obb = o3d.geometry.OrientedBoundingBox(center, rot, extent)
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    color = BOX_COLORS[index % len(BOX_COLORS)]
    line_set.paint_uniform_color(color)
    return line_set


def visualize_scene(path, input_file, score_thresh, display):
    predictions_path = find_predictions_path(path)
    if predictions_path is None:
        raise FileNotFoundError(f'Expected a scene directory or predictions.json, got: {path}')

    scene_dir = predictions_path.parent
    payload, objects = load_predictions(predictions_path, score_thresh=score_thresh)
    full_scene_path = resolve_input_file(scene_dir, payload['frame_id'], input_file)

    geometries = [make_point_cloud(full_scene_path, display['colorless_color'])]
    geometries.extend(make_box_geometry(obj, idx) for idx, obj in enumerate(objects))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f'{payload["frame_id"]}: {len(objects)} boxes')
    render_option = vis.get_render_option()
    render_option.point_size = display['point_size']
    render_option.background_color = display['background_color']
    for geometry in geometries:
        vis.add_geometry(geometry)
    apply_view_options(vis, display, geometries)
    print(f'Opening scene: {full_scene_path}')
    print(f'Boxes: {len(objects)} from {predictions_path}')
    vis.run()
    vis.destroy_window()


def main():
    args = parse_args()
    visualize_scene(
        args.path,
        input_file=args.input_file,
        score_thresh=args.score_thresh,
        display=load_display_options(args),
    )


if __name__ == '__main__':
    main()
