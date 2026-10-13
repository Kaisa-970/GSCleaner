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
parser.add_argument('--targets', nargs='+', type=int, required=True)
parser.add_argument('--mask-root', required=True)
parser.add_argument('--run0', required=True)
parser.add_argument('--run1', required=True)
parser.add_argument('--run2', required=True)
parser.add_argument('--gt', required=True)
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
            'region_definition': 'Target-only white render >0.05 intersect nearest-resized official unseen_mask; estimated support, not semantic GT',
            'train_views': [v.image_name for v in scene.getTrainCameras()],
            'test_views': [v.image_name for v in scene.getTestCameras()],
            'clean_views': [v.image_name for v in scene.getInpaintCameras()]}
background = torch.zeros(3, device='cuda')
views = scene.getInpaintCameras()

import numpy as np
from PIL import Image
from evaluate_regions import rgb, error_record, summarize
records={}
with torch.no_grad():
    for target in args.targets:
        ply=Path(args.run0).parent / f'point_cloud_{target}.ply'
        gaussians.load_ply(str(ply))
        gaussians._features_dc.data.fill_((1.0-0.5)/0.28209479177387814)
        gaussians._features_rest.data.zero_()
        folder=out / str(target);folder.mkdir()
        stats=[]
        for view in views:
            support=render(view,gaussians,pipeline.extract(settings),background)['render'][0].cpu().numpy()>0.05
            with Image.open(Path(args.mask_root)/(view.image_name+'.png')) as mask:
                official=np.array(mask.convert('L').resize((support.shape[1],support.shape[0]),Image.Resampling.NEAREST))>127
            region=support & official
            Image.fromarray(region.astype('uint8')*255).save(folder/(view.image_name+'.png'))
            entry={'view':view.image_name,'support_pixels':int(support.sum()),'eval_pixels':int(region.sum()),'errors':{}}
            for name in ['run1','run2']:
                entry['errors'][name]=error_record(rgb(Path(getattr(args,name))/(view.image_name+'.png')),rgb(Path(args.gt)/(view.image_name+'.png')),region)
            stats.append(entry)
        records[str(target)]={'target_ply_sha256':digest(ply),'views':stats,'aggregate':{name:summarize([v['errors'][name] for v in stats]) for name in ['run1','run2']}}
(out/'targets.json').write_text(json.dumps({'definition':manifest['region_definition'],'targets':records},indent=2,allow_nan=False))
