"""Render black-region evidence; heuristic classes are not ground-truth geometry."""
import argparse
from pathlib import Path
import sys
import json
p=argparse.ArgumentParser()
for name in ('repo','source','model','checkpoint','classifier','remove_mask','output'):
    p.add_argument('--'+name,required=True)
a=p.parse_args(); sys.path.insert(0,a.repo)
import numpy as np
import torch
from PIL import Image, ImageDraw
from arguments import ModelParams, PipelineParams
from scene import Scene
from gaussian_renderer import GaussianModel, render
import inspect
import gaussian_renderer
# Expose rasterizer alpha without changing the deployed renderer.
render_source=inspect.getsource(render)
if render_source.count('"render": rendered_image,') != 1:
    raise RuntimeError('Unsupported renderer return structure')
render_source=render_source.replace('"render": rendered_image,','"render": rendered_image, "alpha": alpha,')
render_globals=dict(vars(gaussian_renderer)); exec(render_source,render_globals)
render=render_globals['render']
defaults=argparse.ArgumentParser(); mp=ModelParams(defaults); pp=PipelineParams(defaults)
s=defaults.parse_args([]); s.source_path=a.source; s.model_path=a.model
s.object_path='associated_hqsam'; s.eval=True; s.resolution=8; s.train_distill=False
g=GaussianModel(3); scene=Scene(mp.extract(s),g,load_iteration=2000,shuffle=False)
g.load_ply(a.checkpoint)
state=torch.load(a.classifier,map_location='cuda',weights_only=True)
c=torch.nn.Conv2d(g.num_objects,state['weight'].shape[0],1).cuda(); c.load_state_dict(state)
with torch.no_grad(): prob=torch.softmax(c(g._objects_dc.permute(2,0,1)),dim=0)[29].reshape(-1)
mask=torch.from_numpy(np.load(a.remove_mask)).cuda().bool()
fields=['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation','_objects_dc']
original={k:getattr(g,k).detach().clone() for k in fields}
out=Path(a.output); out.mkdir(exist_ok=False,parents=True)
summary={'thresholds':{'dark_rgb_mean':0.15,'gt_brightness_excess':0.12,'low_alpha':0.8,'weak_target_probability':0.1},'views':[]}
bg=torch.zeros(3,device='cuda')
def subset(selection):
    for k,v in original.items(): setattr(g,k,torch.nn.Parameter(v[selection],requires_grad=False))
with torch.no_grad():
 for view in scene.getInpaintCameras():
    subset(~mask); r=render(view,g,pp.extract(s),bg)
    rgb=r['render'].detach().cpu().numpy().transpose(1,2,0)
    alpha=r['alpha'].detach().cpu().numpy().squeeze()
    depth=r['depth_3dgs'].detach().cpu().numpy().squeeze()
    subset((~mask)&(prob>0.1)); residual=render(view,g,pp.extract(s),bg)['alpha'].detach().cpu().numpy().squeeze()
    gt=view.original_image.cpu().numpy().transpose(1,2,0)
    dark=(rgb.mean(2)<0.15)&((gt.mean(2)-rgb.mean(2))>0.12)
    # Exclude distant naturally dark regions; diagnose target vicinity independently of GT unseen mask.
    subset(mask); target_alpha=render(view,g,pp.extract(s),bg)['alpha'].detach().cpu().numpy().squeeze()
    dark &= target_alpha>0.05
    hole=dark&(alpha<0.8)
    residual_evidence=dark&(~hole)&(residual>0.1)
    poor_background=dark&(~hole)&(~residual_evidence)
    classes=np.zeros((*dark.shape,3),dtype=np.uint8)
    classes[hole]=[255,0,0]; classes[residual_evidence]=[255,180,0]; classes[poor_background]=[0,150,255]
    folder=out/view.image_name; folder.mkdir()
    for label,m in [('dark',dark),('low_alpha_hole',hole),('possible_residual',residual_evidence),('opaque_background_defect',poor_background)]:
        Image.fromarray(m.astype('uint8')*255).save(folder/(label+'.png'))
    Image.fromarray(classes).save(folder/'classes.png')
    np.savez_compressed(folder/'evidence.npz',alpha=alpha,depth=depth,residual_target_alpha=residual,target_alpha=target_alpha)
    base=(np.clip(rgb,0,1)*255).astype('uint8'); overlay=base.copy(); overlay[dark]=(base[dark]*0.4+classes[dark]*0.6).astype('uint8')
    Image.fromarray(overlay).save(folder/'overlay.png')
    summary['views'].append({'name':view.image_name,'dark':int(dark.sum()),'low_alpha_hole':int(hole.sum()),'possible_residual':int(residual_evidence.sum()),'opaque_background_defect':int(poor_background.sum())})
summary['remaining_target_probability_gt_0_1']=int(((~mask)&(prob>0.1)).sum())
summary['remaining_target_probability_gt_0_7']=int(((~mask)&(prob>0.7)).sum())
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:sum(v[k] for v in summary['views']) for k in ('dark','low_alpha_hole','possible_residual','opaque_background_defect')}))
