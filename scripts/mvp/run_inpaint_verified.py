"""Execute upstream inpainting with a checked deletion-mask gate, no loss changes."""
import os
import sys
import runpy
import json
import time
from pathlib import Path
import numpy as np
import torch

repo=Path(os.environ['MVP_REPO']).resolve()
expected=np.load(os.environ['MVP_REMOVE_MASK']).astype(bool)
report=Path(os.environ['MVP_GATE_REPORT'])
sys.path.insert(0,str(repo))
from scene.gaussian_model import GaussianModel
original=GaussianModel.inpaint_setup
def checked(self,args,opt,masks):
    actual=np.zeros(len(self.get_xyz),dtype=bool)
    for value in masks.values():
        actual |= value['mask3d'].detach().cpu().numpy().reshape(-1).astype(bool)
    same=actual.shape==expected.shape and np.array_equal(actual,expected)
    report.write_text(json.dumps({'exact_match':same,'source_count':len(actual),'removed':int(actual.sum()),'timestamp':time.time()},indent=2))
    if not same:
        raise RuntimeError('Repair deletion mask differs from frozen Run 1; stopped before optimization')
    return original(self,args,opt,masks)
GaussianModel.inpaint_setup=checked
start=time.time()
runpy.run_path(str(repo/'edit_object_inpaint.py'),run_name='__main__')
report.with_name('runtime.json').write_text(json.dumps({'elapsed_seconds':time.time()-start,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'peak_cuda_reserved':torch.cuda.max_memory_reserved()},indent=2))
