from pathlib import Path
import subprocess,sys,os,time,json,hashlib
EXP=Path(__file__).resolve().parents[1]
ARMS=[('rr',42),('control',42),('rr',43),('control',43),('rr',44),('control',44),('dino',42),('r5',42),('m2f',42)]
def main():
    env=dict(os.environ,BAM_STUDY_START=str(time.time()))
    assert not (EXP/'PREDICTIONS_FROZEN.json').exists(),'Preserve completed frozen evaluation'
    log=EXP/'logs';log.mkdir(exist_ok=True)
    for kind,seed in ARMS:
        print('START',kind,seed,flush=True)
        with (log/f'{kind}_{seed}.log').open('a') as f:
            subprocess.run([sys.executable,str(EXP/'code/predict.py'),'--kind',kind,'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',kind,seed,flush=True)
    files=sorted((EXP/'predictions').rglob('b*.json.gz'))
    assert len(files)==216
    hashes={str(p.relative_to(EXP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (EXP/'PREDICTIONS_FROZEN.json').write_text(json.dumps(dict(stage='ALL_RGB_ONLY_PREDICTIONS_FROZEN_BEFORE_REFERENCES',arms=ARMS,tiles=24,forwards=1944,TEST=False,sha256=hashes),indent=2)+'\n')
if __name__=='__main__':main()
