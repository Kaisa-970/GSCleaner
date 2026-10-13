"""Run inside an Inpaint360GS environment; preserve clean-camera image names."""
import sys
from argparse import ArgumentParser
from pathlib import Path
import json
import hashlib

parser = ArgumentParser()
parser.add_argument('--repo', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--resolution', type=int, default=8)
parser.add_argument('--views', choices=('clean', 'holdout'), default='clean')
parser.add_argument('--run0', required=True)
parser.add_argument('--run1', required=True)
parser.add_argument('--run2', required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.repo).resolve()))

import torch
import torchvision
from arguments import ModelParams, PipelineParams
from scene import Scene
from gaussian_renderer import GaussianModel, render

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

defaults = ArgumentParser()
model = ModelParams(defaults)
pipeline = PipelineParams(defaults)
settings = defaults.parse_args([])
settings.source_path = str(Path(args.source).resolve())
settings.model_path = str(Path(args.model).resolve())
settings.object_path = 'associated_hqsam'
settings.resolution = args.resolution
settings.eval = True
settings.train_distill = False
gaussians = GaussianModel(settings.sh_degree)
# Explicit numeric iteration; never inherit train_distill=True from cfg_args.
scene = Scene(model.extract(settings), gaussians, load_iteration=2000, shuffle=False)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
manifest = {'resolution': args.resolution, 'train_distill': False, 'runs': {},
            'rendered_view_set': args.views,
            'train_views': [v.image_name for v in scene.getTrainCameras()],
            'test_views': [v.image_name for v in scene.getTestCameras()],
            'clean_views': [v.image_name for v in scene.getInpaintCameras()]}
background = torch.zeros(3, device='cuda')
views = scene.getInpaintCameras() if args.views == 'clean' else scene.getTestCameras()
with torch.no_grad():
    for label in ('run0', 'run1', 'run2'):
        ply = Path(getattr(args, label)).resolve()
        gaussians.load_ply(str(ply))
        manifest['runs'][label] = {'ply': str(ply), 'sha256': digest(ply),
                                    'count': len(gaussians.get_xyz)}
        folder = out / label
        folder.mkdir()
        for view in views:
            torchvision.utils.save_image(render(view, gaussians, pipeline.extract(settings), background)['render'],
                                         folder / (view.image_name + '.png'))
            gt = out / 'gt'
            gt.mkdir(exist_ok=True)
            if label == 'run0':
                torchvision.utils.save_image(view.original_image, gt / (view.image_name + '.png'))
torch.cuda.synchronize()
manifest['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest['runs'], indent=2))
