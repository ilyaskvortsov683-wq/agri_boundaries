import subprocess, json, sys
V = {
 'V0': {},
 'V1': dict(JUMP_THR=10),
 'V2': dict(JUMP_THR=8),
 'V3': dict(JUMP_THR=10, BARRIER_THR=20),
 'V4': dict(BARRIER_THR=18, BARRIER_MIN_PX=30),
 'V5': dict(JUMP_THR=0, BARRIER_THR=35, BARRIER_MIN_PX=15, BARRIER_DILATE=1, SP_SIZE_PX=120, COMPACTNESS=0.3, MERGE_THR=50),
 'V6': dict(MERGE_THR=38, JUMP_THR=10),
 'V7': dict(SP_SIZE_PX=100, JUMP_THR=10),
 'V8': dict(JUMP_THR=12, BARRIER_THR=22),
 'V9': dict(JUMP_THR=8, BARRIER_THR=20, MERGE_THR=40),
}
for k, v in V.items():
    opt = {"tag": "_P" + k, "over": {"STAGES": {"1": 1, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}, "PRESETS": {"V": v}, "STAGE1_PRESET": "V"}}
    r = subprocess.run(['python3', '-I', 'run_v3.py', json.dumps(opt)], capture_output=True, text=True)
    print(k, v, 'rc', r.returncode, flush=True)
