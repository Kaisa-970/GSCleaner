#!/usr/bin/env python3
"""CPU-only audit of external masks against immutable Gaussian rows.

This produces candidates, never a deletion PLY. Center z-buffer visibility is a
conservative diagnostic, not Gaussian surface depth or semantic ground truth.
Masks MUST share the undistorted COLMAP PINHOLE pixel coordinates.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
try:
    from scipy.ndimage import binary_dilation, binary_erosion, minimum_filter
except ModuleNotFoundError:  # render-only tools need the PLY/COLMAP readers without SciPy.
    binary_dilation = binary_erosion = minimum_filter = None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_ply(path):
    """Memory-map vertex records, retaining exact original row identity."""
    types = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
             'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
             'int': '<i4', 'int32': '<i4', 'uint': '<u4', 'uint32': '<u4',
             'short': '<i2', 'ushort': '<u2'}
    props, count, active, fmt = [], None, False, None
    with Path(path).open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError('not a PLY')
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError('truncated PLY header')
            s = raw.decode('ascii').split()
            if not s:
                continue
            if s[0] == 'format':
                fmt = s[1]
            elif s[0] == 'element':
                active = s[1] == 'vertex'
                if active:
                    count = int(s[2])
            elif s[0] == 'property' and active:
                if s[1] not in types:
                    raise ValueError('unsupported vertex property: ' + ' '.join(s))
                props.append((s[2], types[s[1]]))
            elif s[0] == 'end_header':
                offset = f.tell()
                break
    if fmt != 'binary_little_endian' or count is None:
        raise ValueError('only binary little-endian vertex PLY supported')
    a = np.memmap(path, mode='r', offset=offset, dtype=np.dtype(props), shape=(count,))
    for key in ('x', 'y', 'z', 'opacity'):
        if key not in a.dtype.names:
            raise ValueError('missing Gaussian field: ' + key)
    return a


def read_colmap(sparse):
    cams, views = {}, {}
    with (Path(sparse) / 'cameras.bin').open('rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            cid, model, w, h = struct.unpack('<iiQQ', f.read(24))
            if model != 1:
                raise ValueError('PINHOLE cameras required; undistort masks first')
            cams[cid] = {'width': w, 'height': h, 'params': struct.unpack('<4d', f.read(32))}
    with (Path(sparse) / 'images.bin').open('rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            v = struct.unpack('<i7di', f.read(64))
            name = bytearray()
            while True:
                c = f.read(1)
                if c == b'\0':
                    break
                if not c:
                    raise ValueError('truncated COLMAP name')
                name.extend(c)
            n = struct.unpack('<Q', f.read(8))[0]
            f.seek(n * 24, 1)
            qw, qx, qy, qz = v[1:5]
            norm = np.linalg.norm([qw, qx, qy, qz])
            if not np.isclose(norm, 1, atol=1e-5):
                raise ValueError('non-unit camera quaternion')
            R = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                          [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                          [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
            views[name.decode()] = {'camera_id': v[-1], 'R': R, 't': np.array(v[5:8])}
    return cams, views


def project(xyz, camera, view, reliable, radius):
    p = xyz @ view['R'].T + view['t']
    w, h = camera['width'], camera['height']
    fx, fy, cx, cy = camera['params']
    z = p[:, 2]
    finite = np.isfinite(p).all(axis=1) & (z > .05)
    u = np.zeros(len(p), np.int32)
    v = np.zeros(len(p), np.int32)
    u[finite] = np.rint(fx * p[finite, 0] / z[finite] + cx).astype(np.int32)
    v[finite] = np.rint(fy * p[finite, 1] / z[finite] + cy).astype(np.int32)
    inside = finite & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ids = np.flatnonzero(inside)
    pix = v[ids] * w + u[ids]
    depth = np.full(w * h, np.inf, np.float32)
    good = reliable[ids]
    np.minimum.at(depth, pix[good], z[ids[good]])
    depth = depth.reshape(h, w)
    if radius:
        depth = minimum_filter(depth, size=radius * 2 + 1, mode='constant', cval=np.inf)
    return ids, u[ids], v[ids], z[ids], depth


def resolve(base, name):
    path = Path(name)
    return path if path.is_absolute() else base / path


def audit(args):
    if binary_dilation is None:
        raise RuntimeError('CPU mask lifting needs scipy.ndimage; use the deployed Inpaint360GS environment')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    (args.output / 'manifest_snapshot.json').write_bytes(manifest_bytes)
    specs = manifest.get('views', [])
    if not specs:
        raise ValueError('manifest requires nonempty views')
    cams, views = read_colmap(args.sparse)
    a = read_ply(args.ply)
    xyz = np.column_stack([a[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    opacity = 1 / (1 + np.exp(-np.clip(a['opacity'].astype(np.float64), -80, 80)))
    reliable = np.isfinite(xyz).all(axis=1) & (opacity >= args.min_opacity)
    tolerances = sorted(set([args.depth_abs, *args.sensitivity_depth]))
    counts = {t: {'positive': np.zeros(len(a), np.uint16), 'negative': np.zeros(len(a), np.uint16),
                  'groups': np.zeros(len(a), np.uint16)} for t in tolerances}
    ray_support = np.zeros(len(a), np.uint16)
    group_masks = {t: {} for t in tolerances}
    records = []
    seen = set()
    for spec in specs:
        name = spec['image']
        kind = spec.get('kind', 'target')
        annotation = (name, kind, str(resolve(args.manifest.parent, spec['mask'])))
        if annotation in seen:
            raise ValueError('duplicate annotation: ' + str(annotation))
        if any(previous[:2] == annotation[:2] for previous in seen):
            raise ValueError('merge multiple same-kind annotations per view before lifting: ' + name)
        seen.add(annotation)
        if spec.get('coordinate_space') != 'colmap_pinhole':
            raise ValueError(name + ': mask coordinate_space must explicitly be colmap_pinhole')
        if spec.get('annotation_type', 'pixel_mask') != 'pixel_mask':
            raise ValueError(name + ': boxes/proposals are not target masks')
        if name not in views:
            raise ValueError('view absent from COLMAP: ' + name)
        view = views[name]
        cam = cams[view['camera_id']]
        path = resolve(args.manifest.parent, spec['mask'])
        mask_image = Image.open(path)
        if mask_image.size != (cam['width'], cam['height']):
            raise ValueError(name + ': mask dimensions mismatch; no implicit resize/undistortion allowed')
        target = np.array(mask_image.convert('L')) > args.mask_threshold
        if kind not in ('target', 'negative'):
            raise ValueError('kind must be target or negative')
        # Target mask must describe a complete silhouette for outside-mask negatives.
        negative_outside = bool(spec.get('complete_target_annotation', False)) and kind == 'target'
        inner = binary_erosion(target, iterations=args.boundary_px) if args.boundary_px else target
        outer = binary_dilation(target, iterations=args.boundary_px) if args.boundary_px else target
        ids, u, v, z, depth = project(xyz, cam, view, reliable, args.zbuffer_radius)
        positive_ray = inner[v, u] if kind == 'target' else np.zeros(len(ids), bool)
        negative_ray = inner[v, u] if kind == 'negative' else (~outer[v, u] if negative_outside else np.zeros(len(ids), bool))
        ray_support[ids[positive_ray]] += 1
        per_tol = {}
        for tol in tolerances:
            visible = reliable[ids] & np.isfinite(depth[v, u]) & (z <= depth[v, u] + tol + args.depth_relative * z)
            pos_ids = ids[positive_ray & visible]
            neg_ids = ids[negative_ray & visible]
            counts[tol]['positive'][pos_ids] += 1
            counts[tol]['negative'][neg_ids] += 1
            group = str(spec.get('view_group', Path(name).stem))
            groups = group_masks[tol]
            if group not in groups:
                groups[group] = np.zeros(len(a), bool)
            groups[group][pos_ids] = True
            per_tol[str(tol)] = {'visible_positive_centers': len(pos_ids), 'visible_negative_centers': len(neg_ids)}
        # Show central-depth map and raw positive rays; this is diagnostics, not final selection.
        if args.images:
            base = Image.open(args.images / name).convert('RGB')
            if base.size != mask_image.size:
                raise ValueError('image dimensions mismatch')
        else:
            base = Image.new('RGB', mask_image.size, '#303030')
        overlay = np.array(base).copy()
        overlay[target] = (.5 * overlay[target] + .5 * np.array([255, 180, 0])).astype('uint8')
        tol = args.depth_abs
        visible = reliable[ids] & np.isfinite(depth[v, u]) & (z <= depth[v, u] + tol + args.depth_relative * z)
        overlay[v[positive_ray & visible], u[positive_ray & visible]] = [0, 255, 90]
        finite_depth = np.isfinite(depth)
        image_depth = np.zeros((*depth.shape, 3), np.uint8)
        if finite_depth.any():
            lo, hi = np.percentile(depth[finite_depth], [2, 98])
            gray = np.clip((depth[finite_depth] - lo) / max(hi-lo, 1e-8) * 255, 0, 255).astype('uint8')
            image_depth[finite_depth] = gray[:, None]
        sheet = Image.new('RGB', (cam['width'] * 2, cam['height'] + 30))
        sheet.paste(Image.fromarray(overlay), (0, 30))
        sheet.paste(Image.fromarray(image_depth), (cam['width'], 30))
        ImageDraw.Draw(sheet).text((8, 8), name + ' | amber mask, green visible centers | center z-buffer', fill='white')
        sheet.save(args.output / (name.replace('/', '_').rsplit('.', 1)[0] + '_' + kind + '_audit.jpg'))
        records.append({'image': name, 'camera_id': view['camera_id'], 'kind': kind,
                        'mask': str(path), 'mask_sha256': sha256(path), 'mask_pixels': int(target.sum()),
                        'complete_target_annotation': negative_outside, 'positive_ray_centers': int(positive_ray.sum()),
                        'tolerances': per_tol})
        print(name, per_tol, flush=True)
    sensitivity = []
    for tol in tolerances:
        c = counts[tol]
        for group in group_masks[tol].values():
            c['groups'] += group
        ratio = c['positive'] / np.maximum(c['positive'].astype(float) + c['negative'], 1)
        for min_groups in sorted(set([args.min_view_groups, 2, 3])):
            candidate = (c['positive'] >= args.min_positive) & (c['groups'] >= min_groups) & (c['negative'] <= args.max_negative) & (ratio >= args.min_positive_ratio)
            filename = 'candidate_depth_' + format(tol, '.3f').replace('.', '') + '_groups' + str(min_groups) + '_indices.npy'
            np.save(args.output / filename, np.flatnonzero(candidate))
            sensitivity.append({'depth_abs_m': tol, 'min_view_groups': min_groups, 'candidates': int(candidate.sum()),
                                'indices_file': filename, 'indices_sha256': sha256(args.output / filename)})
    c = counts[args.depth_abs]
    ratio = c['positive'] / np.maximum(c['positive'].astype(float) + c['negative'], 1)
    candidate = (c['positive'] >= args.min_positive) & (c['groups'] >= args.min_view_groups) & (c['negative'] <= args.max_negative) & (ratio >= args.min_positive_ratio)
    uncertain = (ray_support > 0) & ~candidate
    np.save(args.output / 'candidate_indices.npy', np.flatnonzero(candidate))
    np.save(args.output / 'uncertain_indices.npy', np.flatnonzero(uncertain))
    np.savez_compressed(args.output / 'support_counts.npz', positive=c['positive'], negative=c['negative'],
                        positive_groups=c['groups'], mask_ray_support=ray_support)
    bbox = lambda m: None if not m.any() else {'min': xyz[m].min(axis=0).tolist(), 'max': xyz[m].max(axis=0).tolist()}
    report = {'status': 'candidate_not_yet_visually_verified', 'ply': str(args.ply),
              'asset_sha256': sha256(args.ply), 'source_count': len(a), 'manifest_sha256': manifest_hash,
              'cameras_sha256': sha256(args.sparse / 'cameras.bin'), 'images_sha256': sha256(args.sparse / 'images.bin'),
              'rules': {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
              'candidate_count': int(candidate.sum()), 'uncertain_count': int(uncertain.sum()),
              'candidate_bbox': bbox(candidate), 'uncertain_bbox': bbox(uncertain), 'views': records,
              'threshold_sensitivity': sensitivity,
              'limitations': ['Center z-buffer is not a Gaussian rendered surface or strict occlusion test.',
                              'Masks, candidate calibration and opacity threshold can bias membership.',
                              'Stereo frames with identical stem count as one group; temporal separation still needs human review.',
                              'Outside-mask negative evidence is only enabled for complete_target_annotation=true.',
                              'No convex hull, no deletion PLY, no semantic zero-false-deletion claim.',
                              'Thin forks, chains, translucent/low-opacity points and hidden vehicle parts may be missed.',
                              'Remote wall centers along mask rays are excluded unless near the center z-buffer, but this cannot certify background protection.']}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('candidate', int(candidate.sum()), 'uncertain', int(uncertain.sum()), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ply', type=Path, required=True)
    p.add_argument('--sparse', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--images', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mask-threshold', type=int, default=127)
    p.add_argument('--boundary-px', type=int, default=2)
    p.add_argument('--min-opacity', type=float, default=.05)
    p.add_argument('--zbuffer-radius', type=int, default=1)
    p.add_argument('--depth-abs', type=float, default=.08)
    p.add_argument('--depth-relative', type=float, default=.01)
    p.add_argument('--sensitivity-depth', type=float, nargs='+', default=[.03, .08, .20])
    p.add_argument('--min-positive', type=int, default=2)
    p.add_argument('--min-view-groups', type=int, default=2)
    p.add_argument('--max-negative', type=int, default=0)
    p.add_argument('--min-positive-ratio', type=float, default=.8)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if min(args.boundary_px, args.zbuffer_radius, args.depth_abs, args.depth_relative) < 0:
        raise ValueError('negative geometric tolerance')
    audit(args)
