#!/usr/bin/env python3
"""SAM point/box proposals for the Wanwei7 forklift.

This deliberately produces 2-D, COLMAP-PINHOLE masks only.  It never edits a
PLY or infers a 3-D deletion set.  Each view has several component prompts so
that thin mast/fork structures are retained while background is excluded.
"""
import argparse, json
from pathlib import Path
import cv2, numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--image-root',required=True)
    ap.add_argument('--output',required=True); ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--repo',required=True); ap.add_argument('--gpu',default='cuda'); ap.add_argument('--views',required=True)
    args=ap.parse_args()
    import sys; sys.path.insert(0,args.repo)
    from tool.segmentor import Segmentor
    from model_args import sam_args
    cfg=dict(sam_args); cfg['sam_checkpoint']=args.checkpoint; cfg['gpu_id']=args.gpu
    seg=Segmentor(cfg); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    specs=json.loads(Path(args.views).read_text())
    records=[]
    for spec in specs:
        rel=spec['image']; src=Path(args.image_root)/rel
        bgr=cv2.imread(str(src),cv2.IMREAD_COLOR)
        if bgr is None: raise FileNotFoundError(src)
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); seg.have_embedded=False; seg.set_image(rgb)
        stem=Path(rel).with_suffix('').as_posix().replace('/','_')
        union=np.zeros(rgb.shape[:2],np.uint8); parts=[]
        for part in spec['parts']:
            pts=np.asarray(part['points'],np.float32); labels=np.asarray(part.get('labels',[1]*len(pts)),np.int32)
            # A component box is a hard safety fence.  The SAM prediction is
            # still guided by positive/negative points, but cannot consume a
            # wall or floor outside the annotated component.
            box=np.asarray(part.get('box'),np.float32) if part.get('box') else None
            masks,scores,_=seg.interactive_predictor.predict(point_coords=pts,point_labels=labels,box=box,multimask_output=True)
            idx=int(np.argmax(scores)); m=masks[idx].astype(np.uint8)
            # A component must contain its positive seed. Reject implausible full-frame choices.
            positives=pts[labels==1].astype(int)
            if len(positives) and not all(m[max(0,min(m.shape[0]-1,y)),max(0,min(m.shape[1]-1,x))] for x,y in positives):
                idx=int(np.argmax([mm[tuple(np.clip(positives[:,1],0,m.shape[0]-1)),tuple(np.clip(positives[:,0],0,m.shape[1]-1))].mean() for mm in masks]))
                m=masks[idx].astype(np.uint8)
            if box is not None:
                gate=np.zeros_like(m); x1,y1,x2,y2=box.astype(int);gate[max(0,y1):min(m.shape[0],y2+1),max(0,x1):min(m.shape[1],x2+1)]=1;m*=gate
            if part.get('polygon'):
                gate=np.zeros_like(m);cv2.fillPoly(gate,[np.asarray(part['polygon'],np.int32)],1);m*=gate
            for hole in part.get('exclude_polygons',[]): cv2.fillPoly(m,[np.asarray(hole,np.int32)],0)
            cv2.imwrite(str(out/(stem+'_'+part['name']+'_component.png')),m*255)
            union=np.maximum(union,m); parts.append({'name':part['name'],'points':part['points'],'labels':part.get('labels',[1]*len(pts)),'score':float(scores[idx]),'area':int(m.sum())})
        # Remove fisheye border and retain a reproducible raster mask.
        union[:2]=0; union[-2:]=0; union[:,:2]=0; union[:,-2:]=0
        mp=out/(stem+'_mask.png'); pp=out/(stem+'_preview.jpg')
        cv2.imwrite(str(mp),union*255)
        ov=rgb.copy(); ov[union>0]=(0.25*ov[union>0]+0.75*np.array([0,255,0])).astype(np.uint8)
        cv2.imwrite(str(pp),cv2.cvtColor(ov,cv2.COLOR_RGB2BGR))
        records.append({'image':rel,'mask':mp.name,'kind':'target','coordinate_space':'colmap_pinhole','complete_target_annotation':False,'shape':list(union.shape),'area':int(union.sum()),'fraction':float(union.mean()),'parts':parts,'prompt_spec':spec})
    (out/'manifest.json').write_text(json.dumps({'coordinate_space':'colmap_pinhole','views':records},indent=2))

if __name__=='__main__': main()
