"""Run one experiment step and sample its process-tree GPU memory via nvidia-smi."""
import argparse, datetime, json, os, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output',required=True,type=Path)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if not cmd:p.error('command required')
a.output.parent.mkdir(parents=True,exist_ok=True)
start=time.time();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
proc=subprocess.Popen(cmd,start_new_session=True)
samples=[];errors=[]
while proc.poll() is None:
    try:
        # A new session distinguishes this step and all descendants from UI/other jobs.
        listing=subprocess.check_output(['ps','-eo','pid=,sid='],text=True)
        ids={int(x.split()[0]) for x in listing.splitlines() if int(x.split()[1])==proc.pid}
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
        used=sum(int(x.split(',')[1]) for x in gpu.splitlines() if int(x.split(',')[0]) in ids)
        samples.append({'elapsed_seconds':time.time()-start,'process_gpu_mib':used})
    except (subprocess.SubprocessError,ValueError) as e:errors.append(str(e))
    time.sleep(.5)
code=proc.wait()
a.output.write_text(json.dumps({'command':cmd,'started_utc':stamp,'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.time()-start,'exit_code':code,'peak_sampled_process_gpu_mib':max((x['process_gpu_mib'] for x in samples),default=None),'sample_interval_seconds':.5,'samples':samples,'measurement_errors':errors,'limitations':'Sampled process-tree GPU usage, not instantaneous peak; may miss spikes between samples.'},indent=2))
raise SystemExit(code)
