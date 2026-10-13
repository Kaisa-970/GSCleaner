#!/usr/bin/env python3
"""Evaluate RGB errors on actual mask pixels, never on zero-filled images.

Requires numpy and Pillow. Example:
  python evaluate_regions.py --run0 run0/renders --run1 run1/renders \
    --run2 run2/renders --gt data/inpaint360/bag/images \
    --masks data/inpaint360/bag/unseen_mask --output results/regions

By default GT selection is test_*; use --gt-glob '*' for a GT-only directory.
White (>127) mask pixels are the repair region by default. GT and render RGB
sizes must match. Mask resizing requires --resize-masks and uses nearest-neighbor.
PSNR uses normalized [0,1] RGB MSE; 'inf' denotes an exact match, null an empty
region. No LPIPS/SSIM/FID claims are made by this script.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


def image_index(directory, pattern='*'):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'Not an image directory: {directory}')
    result = {}
    for path in sorted(directory.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        if path.stem in result:
            raise ValueError(f'Ambiguous image stem {path.stem}: {result[path.stem]}, {path}')
        result[path.stem] = path
    return result


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float64) / 255.0


def psnr(mse):
    if mse is None:
        return None
    return 'inf' if mse == 0 else -10.0 * math.log10(mse)


def error_record(a, b, region):
    count = int(np.count_nonzero(region))
    if count == 0:
        return {'pixels': 0, 'sse_rgb': 0.0, 'mse_rgb': None, 'psnr_db': None}
    residual = a[region] - b[region]
    sse = float(np.sum(residual * residual))
    mse = sse / (3 * count)
    return {'pixels': count, 'sse_rgb': sse, 'mse_rgb': mse, 'psnr_db': psnr(mse)}


def summarize(records):
    nonempty = [r for r in records if r['pixels'] > 0]
    count = sum(r['pixels'] for r in records)
    sse = sum(r['sse_rgb'] for r in records)
    mse = sse / (3 * count) if count else None
    view_mse = float(np.mean([r['mse_rgb'] for r in nonempty])) if nonempty else None
    # Mean PSNR differs from PSNR of mean MSE. Preserve perfect views explicitly.
    scores = [r['psnr_db'] for r in nonempty]
    view_psnr = ('inf' if 'inf' in scores else float(np.mean(scores))) if scores else None
    return {
        'views': len(records), 'nonempty_views': len(nonempty), 'pixels': count,
        'pixel_weighted_mse_rgb': mse, 'pixel_weighted_psnr_db': psnr(mse),
        'view_mean_mse_rgb': view_mse, 'view_mean_psnr_db': view_psnr,
    }


def evaluate(args):
    indexes = {
        'gt': image_index(args.gt, args.gt_glob),
        'mask': image_index(args.masks),
        **{name: image_index(getattr(args, name)) for name in ('run0', 'run1', 'run2')},
    }
    stems = sorted(indexes['gt'])
    if not stems:
        raise ValueError(f'No GT images match {args.gt_glob!r} in {args.gt}')
    for name, index in indexes.items():
        missing = sorted(set(stems) - set(index))
        if missing:
            raise ValueError(f'{name} missing {len(missing)} GT stems: {missing[:10]}')
    comparisons = [('run0', 'gt'), ('run1', 'gt'), ('run2', 'gt'),
                   ('run2', 'run1'), ('run1', 'run0'), ('run2', 'run0')]
    rows, views, resized = [], [], []
    aggregates = {}
    for stem in stems:
        images = {name: rgb(indexes[name][stem]) for name in ('gt', 'run0', 'run1', 'run2')}
        shape = images['gt'].shape
        for name, array in images.items():
            if array.shape != shape:
                raise ValueError(f'{stem}: {name} shape {array.shape} != GT {shape}; RGB resize forbidden')
        mask_path = indexes['mask'][stem]
        with Image.open(mask_path) as image:
            image = image.convert('L')
            original_size = image.size
            desired_size = (shape[1], shape[0])
            if image.size != desired_size:
                if not args.resize_masks:
                    raise ValueError(f'{stem}: mask {image.size} != RGB {desired_size}; pass --resize-masks explicitly')
                image = image.resize(desired_size, Image.Resampling.NEAREST)
                resized.append({'stem': stem, 'from_wh': list(original_size), 'to_wh': list(desired_size)})
            mask = np.asarray(image) > args.mask_threshold
        if args.mask_polarity == 'black':
            mask = ~mask
        regions = {'repair': mask, 'nonrepair': ~mask, 'full': np.ones(mask.shape, dtype=bool)}
        view = {'stem': stem, 'size_wh': list(desired_size),
                'repair_fraction': float(np.mean(mask)),
                'mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                'files': {name: str(index[stem].resolve()) for name, index in indexes.items()},
                'comparisons': {}}
        for left, right in comparisons:
            comparison = f'{left}_vs_{right}'
            view['comparisons'][comparison] = {}
            for region, selection in regions.items():
                record = error_record(images[left], images[right], selection)
                view['comparisons'][comparison][region] = record
                rows.append({'stem': stem, 'comparison': comparison, 'region': region,
                             'repair_fraction': view['repair_fraction'], **record})
                aggregates.setdefault(comparison, {}).setdefault(region, []).append(record)
        views.append(view)
    return {
        'schema_version': 1,
        'metric_definition': 'Normalized RGB SSE / (3 * selected pixel count); PSNR=-10log10(MSE)',
        'interpretation': 'Run-to-run error measures change, not correctness; clean GT comparisons measure reference agreement.',
        'configuration': {
            'directories': {name: str(Path(value).resolve()) for name, value in
                            [('gt', args.gt), ('mask', args.masks), ('run0', args.run0),
                             ('run1', args.run1), ('run2', args.run2)]},
            'gt_glob': args.gt_glob, 'mask_threshold': args.mask_threshold,
            'mask_polarity': args.mask_polarity, 'mask_resize': 'nearest' if args.resize_masks else 'forbidden',
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        'ignored_extra_stems': {name: sorted(set(index) - set(stems)) for name, index in indexes.items()},
        'mask_resizes': resized,
        'aggregate': {comparison: {region: summarize(records) for region, records in regions.items()}
                      for comparison, regions in aggregates.items()},
        'views': views,
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('run0', 'run1', 'run2', 'gt', 'masks'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    parser.add_argument('--gt-glob', default='test_*', help='GT image selection pattern (default: test_*)')
    parser.add_argument('--mask-threshold', type=int, default=127)
    parser.add_argument('--mask-polarity', choices=('white', 'black'), default='white')
    parser.add_argument('--resize-masks', action='store_true')
    parser.add_argument('--output', required=True, type=Path, help='Output directory for regions.json and regions.csv')
    args = parser.parse_args()
    if not 0 <= args.mask_threshold <= 254:
        parser.error('--mask-threshold must be between 0 and 254')
    try:
        report, rows = evaluate(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Evaluation failed: {exc}\n')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'regions.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (args.output / 'regions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Evaluated {len(report["views"])} clean-reference views; results in {args.output}')


if __name__ == '__main__':
    main()
