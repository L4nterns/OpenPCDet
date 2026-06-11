#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(description='Crop labelCloud prediction boxes from full-scene point clouds.')
    parser.add_argument('path', help='A scene prediction directory or predictions.json.')
    parser.add_argument('--input-file', default=None, help='Original full-scene .pcd. Needed only when it cannot be resolved from summary.json.')
    parser.add_argument('--output-dir', default=None, help='Crop output directory. Defaults to output/crops/<cfg>/<tag>/<scene> when possible.')
    parser.add_argument('--score-thresh', type=float, default=None, help='Optional score filter for cropped boxes.')
    return parser.parse_args()


def load_points(path):
    path = Path(path)
    if path.suffix.lower() == '.pcd':
        pcd = o3d.io.read_point_cloud(str(path), remove_nan_points=True, remove_infinite_points=True)
        xyz = np.asarray(pcd.points, dtype=np.float64)
        colors = np.asarray(pcd.colors, dtype=np.float64)
        if colors.shape == xyz.shape and colors.size > 0:
            return xyz, np.clip(colors, 0.0, 1.0)
        return xyz, None

    raise ValueError(f'Unsupported point cloud extension: {path}')


def find_predictions_path(path):
    path = Path(path)
    if path.is_file() and path.name == 'predictions.json':
        return path
    if path.is_dir() and (path / 'predictions.json').exists():
        return path / 'predictions.json'
    raise FileNotFoundError(f'Expected a scene directory or predictions.json, got: {path}')


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


def default_output_dir(scene_dir):
    parts = scene_dir.parts
    if 'predictions' in parts:
        idx = parts.index('predictions')
        return Path(*parts[:idx], 'crops', *parts[idx + 1:])
    return scene_dir


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value)


def crop_filename(obj, suffix):
    return f"{int(obj['index']):03d}_{safe_name(obj['name'])}_{float(obj['score']):.4f}.{suffix}"


def box_from_object(obj):
    box = obj['box']
    return np.array([
        box['x'], box['y'], box['z'],
        box['dx'], box['dy'], box['dz'],
        box['heading'],
    ], dtype=np.float64)


def points_in_rotated_box(points_xyz, box):
    center = box[:3]
    dims = box[3:6]
    heading = box[6]
    shifted = points_xyz - center
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    local_x = cos_h * shifted[:, 0] + sin_h * shifted[:, 1]
    local_y = -sin_h * shifted[:, 0] + cos_h * shifted[:, 1]
    local_z = shifted[:, 2]
    eps = 1e-6
    return (
        (np.abs(local_x) <= dims[0] / 2.0 + eps) &
        (np.abs(local_y) <= dims[1] / 2.0 + eps) &
        (np.abs(local_z) <= dims[2] / 2.0 + eps)
    )


def save_pcd(path, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + '.tmp' + path.suffix)
    if not o3d.io.write_point_cloud(str(tmp_path), pcd, write_ascii=False, compressed=False):
        raise RuntimeError(f'Failed to write PCD: {tmp_path}')
    tmp_path.replace(path)


def save_crop(output_dir, obj, points, colors):
    box = box_from_object(obj)
    mask = points_in_rotated_box(points, box)
    crop_points = points[mask]
    crop_colors = colors[mask] if colors is not None else None

    save_pcd(output_dir / crop_filename(obj, 'pcd'), crop_points, crop_colors)
    return int(crop_points.shape[0])


def main():
    args = parse_args()
    predictions_path = find_predictions_path(args.path)
    scene_dir = predictions_path.parent
    payload, objects = load_predictions(predictions_path, score_thresh=args.score_thresh)
    input_file = resolve_input_file(scene_dir, payload['frame_id'], args.input_file)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(scene_dir)
    points, colors = load_points(input_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for obj in objects:
        num_points = save_crop(output_dir, obj, points, colors)
        rows.append({
            'index': int(obj['index']),
            'name': obj['name'],
            'score': float(obj['score']),
            'num_points': num_points,
        })

    summary = {
        'frame_id': payload['frame_id'],
        'predictions': predictions_path.as_posix(),
        'input_file': Path(input_file).as_posix(),
        'output_dir': output_dir.as_posix(),
        'objects': rows,
    }
    summary_path = output_dir / 'crop_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f'Exported crops: {len(rows)} to {output_dir}')


if __name__ == '__main__':
    main()
