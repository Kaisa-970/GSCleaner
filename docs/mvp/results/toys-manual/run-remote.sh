#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/ypk/Projects/Inpaint360GS
source env.sh
TASK_ROOT="$PWD/output/mvp-toys-manual-20261013"
TASK_SOURCE="$TASK_ROOT/data/inpaint360/toys"
TASK_MODEL="$TASK_ROOT/model"
export TASK_ROOT TASK_SOURCE TASK_MODEL
python - <<'PY' > "$TASK_ROOT/prepare.log" 2>&1
import os,shutil
from pathlib import Path
from argparse import Namespace
import tools.prepare_lama_data as p
r=Path(os.environ['TASK_ROOT'])
def frozen(input_dir,output_dir,*args,**kwargs):
 shutil.copytree(r/'processed_masks',output_dir,dirs_exist_ok=True)
p.enlarge=frozen
p.copy_inpaint2lama_virtual(Namespace(source_path=os.environ['TASK_SOURCE'],model_path=os.environ['TASK_MODEL'],resolution=8,iterations=2000))
PY
(
cd LaMa
export TORCH_HOME="$PWD" PYTHONPATH="$PWD"
python bin/predict_color.py --data_name 360_toys_virtual > "$TASK_ROOT/lama-color.log" 2>&1
python bin/predict_depth.py --data_name 360_toys_virtual > "$TASK_ROOT/lama-depth.log" 2>&1
)
source env.sh
python tools/prepare_lama_data.py -s "$TASK_SOURCE" -m "$TASK_MODEL" -r 8 > "$TASK_ROOT/copy.log" 2>&1
python edit_object_removal_plyfusion.py -s "$TASK_SOURCE" -m "$TASK_MODEL" --config_file "$TASK_ROOT/config/object_removal/toys.json" > "$TASK_ROOT/fusion.log" 2>&1
export MVP_REPO="$PWD" MVP_REMOVE_MASK="$PWD/output/mvp-toys-20261013/removal/remove_mask.npy" MVP_GATE_REPORT="$TASK_ROOT/deletion_gate.json"
python /tmp/run_inpaint_verified.py -s "$TASK_SOURCE" -m "$TASK_MODEL" --config_file "$TASK_ROOT/config/object_inpaint/toys.json" --resolution 8 --render_video > "$TASK_ROOT/inpaint.log" 2>&1
printf 'complete\n' > "$TASK_ROOT/STATUS"
