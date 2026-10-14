"""Diagnostic seed-derived box plus multi-view silhouette candidates; no PLY export."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from lift_forklift_masks import read_ply,read_colmap,sha256
from scipy.ndimage import binary_erosion
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lift-output', type=Path, required=True)
parser.add_argument('--sparse', type=Path, required=True)
parser.add_argument('--manifest-base', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--bbox-min', type=float, nargs=3)
parser.add_argument('--bbox-max', type=float, nargs=3)
args=parser.parse_args()
root=args.lift_output
out=args.output;out.mkdir(parents=True, exist_ok=False)
r=json.loads((root/'report.json').read_text()); manifest=json.loads((root/'manifest_snapshot.json').read_text())
ply=Path(r['ply'])
if sha256(ply)!=r['asset_sha256']:raise ValueError('source PLY changed since lifting')
if sha256(root/'manifest_snapshot.json')!=r['manifest_sha256']:raise ValueError('manifest snapshot hash mismatch')
for key,filename in [('cameras_sha256','cameras.bin'),('images_sha256','images.bin')]:
 if sha256(args.sparse/filename)!=r[key]:raise ValueError('COLMAP calibration changed since lifting: '+filename)
a=read_ply(ply)
if len(a)!=r['source_count']:raise ValueError('source row count mismatch')
xyz=np.column_stack([a[k] for k in ('x','y','z')]).astype(np.float64)
seedpath=root/'candidate_depth_0030_groups3_indices.npy'
entry=next(v for v in r['threshold_sensitivity'] if v['indices_file']==seedpath.name)
if sha256(seedpath)!=entry['indices_sha256']:raise ValueError('seed indices hash mismatch')
seed=np.load(seedpath)
if seed.dtype.kind not in 'iu' or len(seed)!=entry['candidates'] or np.any(seed<0) or np.any(seed>=len(a)) or np.any(np.diff(seed)<=0):raise ValueError('invalid exact source seed indices')
low,high=np.percentile(xyz[seed],[1,99],axis=0);low-=.15;high+=.15
if bool(args.bbox_min) != bool(args.bbox_max):raise ValueError('explicit bounds need both min and max')
if args.bbox_min:
 low=np.array(args.bbox_min);high=np.array(args.bbox_max)
 if np.any(low>=high):raise ValueError('invalid explicit envelope bounds')
inside=np.isfinite(xyz).all(axis=1)&(xyz>=low).all(axis=1)&(xyz<=high).all(axis=1);ids=np.flatnonzero(inside)
cams,views=read_colmap(args.sparse)
groups={};negative=np.zeros(len(ids),np.uint16);positive=np.zeros(len(ids),np.uint16);records=[]
base=args.manifest_base
for spec in manifest['views']:
 name=spec['image'];view=views[name];cam=cams[view['camera_id']];p=xyz[ids]@view['R'].T+view['t'];z=p[:,2];fx,fy,cx,cy=cam['params'];valid=z>.05
 u=np.zeros(len(ids),np.int32);v=u.copy();u[valid]=np.rint(fx*p[valid,0]/z[valid]+cx).astype(np.int32);v[valid]=np.rint(fy*p[valid,1]/z[valid]+cy).astype(np.int32)
 valid&=(u>=0)&(u<cam['width'])&(v>=0)&(v<cam['height']);path=Path(spec['mask']);path=path if path.is_absolute() else base/path
 expected=next(record['mask_sha256'] for record in r['views'] if record['image']==name and record['kind']==spec.get('kind','target'))
 if sha256(path)!=expected:raise ValueError('mask changed since lifting: '+str(path))
 mask_image=Image.open(path)
 if mask_image.size!=(cam['width'],cam['height']):raise ValueError('mask dimensions mismatch')
 mask=np.array(mask_image.convert('L'))>127;mask=binary_erosion(mask,iterations=2);hit=np.zeros(len(ids),bool);hit[valid]=mask[v[valid],u[valid]]
 kind=spec.get('kind','target')
 if kind=='negative':negative+=hit
 else:
  positive+=hit;g=str(spec.get('view_group',Path(name).stem));groups.setdefault(g,np.zeros(len(ids),bool));groups[g]|=hit
 records.append({'image':name,'kind':kind,'mask_sha256':sha256(path),'in_envelope_ray_hits':int(hit.sum())})
count=sum(groups.values(),np.zeros(len(ids),np.uint16));variants=[]
for label,selected in [('all_envelope',ids),('envelope_negative_veto',ids[negative==0])]:
 fn=label+'_indices.npy';np.save(out/fn,selected);variants.append({'selection':label,'candidate_count':len(selected),'indices_file':fn,'indices_sha256':sha256(out/fn)})
for n in (2,3):
 selected=ids[(count>=n)&(positive>=2)&(negative==0)];fn=f'candidate_groups{n}_indices.npy';np.save(out/fn,selected);hidden=np.setdiff1d(selected,seed,assume_unique=True);np.save(out/f'hidden_or_not_strong_seed_groups{n}_indices.npy',hidden)
 variants.append({'min_positive_groups':n,'candidate_count':len(selected),'indices_file':fn,'indices_sha256':sha256(out/fn),'not_strong_seed_count':len(hidden)})
np.savez_compressed(out/'support_counts.npz',source_indices=ids,positive=positive,negative=negative,positive_groups=count)
report={'status':'candidate_not_yet_visually_verified','source_ply':str(ply),'asset_sha256':r['asset_sha256'],'source_count':len(a),'seed_file':str(root/'candidate_depth_0030_groups3_indices.npy'),'seed_sha256':sha256(root/'candidate_depth_0030_groups3_indices.npy'),'manifest_sha256':r['manifest_sha256'],'cameras_sha256':sha256(args.sparse/'cameras.bin'),'images_sha256':sha256(args.sparse/'images.bin'),'rules':{'envelope':('explicit axis-aligned diagnostic bounds' if args.bbox_min else 'axis-aligned seed 1st/99th percentile plus 0.15m on each axis'),'bbox_min':low.tolist(),'bbox_max':high.tolist(),'projection':'COLMAP PINHOLE Gaussian centers; all positive depths, no center zbuffer visibility gate','positive_mask_erosion_px':2,'negative_mask_erosion_px':2,'negative_veto':'any explicit background ROI projection, regardless of depth','positive_min_observations':2,'minimum_groups':[2,3]},'envelope_row_count':len(ids),'explicit_negative_veto_count':int((negative>0).sum()),'variants':variants,'views':records,'limitations':['Seed-derived envelope has not been established to include the entire vehicle or exclude background.','Unlimited depth silhouette intersections may contain background within envelope.','Explicit background projection veto is conservative and may reject vehicle behind a foreground background ROI.','Partial annotations do not support outside-mask negatives.','Non-strong-seed indices are diagnostically hidden/weakly-visible, not proven vehicle membership.','Exact source rows do not imply exact semantic membership. No final deletion PLY exported.']}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
