from pathlib import Path
import json,gzip,csv,hashlib
EXP=Path(__file__).resolve().parents[1];REPO=EXP.parents[1]
CONTROL=REPO/'research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/scale_exposure_control_v1'
PILOT=CONTROL.parent/'frozen_encoding_v1'
OLD=REPO/'research/ITC_MASK_DINO_FORMATION_V1_20261104'
def read(p):return json.loads(Path(p).read_text())
def gz(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
    if p.suffix=='.gz':
        with gzip.open(p,'wt') as f:json.dump(d,f,allow_nan=False)
    else:p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def csvwrite(p,rs):
    ks=list(dict.fromkeys(k for r in rs for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=ks);w.writeheader();w.writerows(rs)
