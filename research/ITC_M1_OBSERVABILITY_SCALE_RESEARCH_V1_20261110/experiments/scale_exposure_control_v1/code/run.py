"""Sequential two-arm execution; one live handle, no hidden scheduler."""
from pathlib import Path
import subprocess,sys,json
EXP=Path(__file__).resolve().parents[1]
for arm in ['S0','S1']:
    status=EXP/'logs'/arm/'STATUS.json'
    if status.exists() and json.loads(status.read_text())['stage']=='COMPLETED_FIXED_SCHEDULE':continue
    print('START',arm,flush=True)
    subprocess.run([sys.executable,str(EXP/'code/train.py'),'--arm',arm],check=True)
    print('DONE',arm,flush=True)
