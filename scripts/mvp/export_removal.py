"""Instrument a copy of upstream removal_setup; never overwrite deployed outputs."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--classifier', required=True)
p.add_argument('--reference')
p.add_argument('--output', required=True)
p.add_argument('--ids', type=int, nargs='+', required=True)
p.add_argument('--threshold', type=float, default=0.7)
args = p.parse_args()
sys.path.insert(0, str(Path(args.repo).resolve()))
import numpy as np
import torch
from plyfile import PlyData
import edit_object_removal as upstream
from gaussian_renderer import GaussianModel

out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=False)
def save_masks(obj_id, threshold, hull, final):
    for name, mask in [('threshold', threshold), ('hull', hull), ('final', final)]:
        np.save(out / f'object_{obj_id}_{name}.npy', mask.detach().cpu().numpy().reshape(-1).astype(bool))

source = inspect.getsource(upstream.removal_setup)
needle = '            mask3d = torch.logical_or(mask3d,mask3d_convex)'
if source.count(needle) != 1:
    raise RuntimeError('Unsupported upstream removal_setup; inspect before instrumenting')
source = source.replace(needle, '            save_masks(obj_id_int, mask3d, mask3d_convex, torch.logical_or(mask3d, mask3d_convex))\n' + needle)
namespace = dict(vars(upstream), save_masks=save_masks)
exec(compile(source, '<instrumented removal_setup>', 'exec'), namespace)
gaussians = GaussianModel(3)
gaussians.load_ply(args.checkpoint)
state = torch.load(args.classifier, map_location='cuda', weights_only=True)
classifier = torch.nn.Conv2d(gaussians.num_objects, state['weight'].shape[0], 1).cuda()
classifier.load_state_dict(state)
classifier.eval()
original_count = len(gaussians.get_xyz)
namespace['removal_setup'](str(out), 2000, gaussians, classifier, args.ids, args.threshold)
masks = [np.load(out / f'object_{i}_final.npy') for i in args.ids]
union = np.logical_or.reduce(masks)
np.save(out / 'remove_mask.npy', union)
np.save(out / 'remove_indices.npy', np.flatnonzero(union))
# Verify every property and row, not coordinate matching or merely point counts.
original = PlyData.read(args.checkpoint)['vertex'].data
reference = PlyData.read(args.reference)['vertex'].data if args.reference else None
generated_path = out / 'point_cloud_object_removal/iteration_2000/point_cloud.ply'
generated = PlyData.read(str(generated_path))['vertex'].data
kept = original[~union]
exact = generated.dtype == kept.dtype and np.array_equal(kept, generated)
if reference is not None:
    exact = exact and reference.dtype == generated.dtype and np.array_equal(reference, generated)
report = {'ids': args.ids, 'threshold': args.threshold, 'original_count': original_count,
          'removed_count': int(union.sum()), 'kept_count': len(generated),
          'reference_exact_all_properties': bool(exact) if reference is not None else None,
          'source_subset_exact_all_properties': bool(exact),
          'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
          'upstream_function_sha256': hashlib.sha256(inspect.getsource(upstream.removal_setup).encode()).hexdigest(),
          'peak_cuda_bytes': torch.cuda.max_memory_allocated()}
(out / 'removal_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if not exact:
    raise RuntimeError('Existing Run 1 is not exactly reproduced; no controlled comparison claimed')
