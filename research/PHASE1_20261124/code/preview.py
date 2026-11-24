"""Preview: M2 (seed 42) REFORM examples on validation. 8 random REFORM-labelled candidates (seed 0).
Each 320px panel: yellow = GT outlines; red = parent candidate; cyan/magenta = M2's two children.
Title: whether M2 decided REFORM, and whether a child reached IoU>=0.5 with a GT. Coarse eye check only."""
import json, sys
from pathlib import Path
import numpy as np, cv2, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from common import crops, load_maskrcnn, maskrcnn_forward, roi_box, crop_resample, roi_features, candidate_labels, STUDY
from arms import ARMS
from evaluate import children_M2
shift = json.load(open(STUDY/"results/EVAL.json"))["prior_shift"]
m = ARMS["M2"]().cuda().eval(); m.load_state_dict(torch.load(STUDY/"results/runs/seed42/M2.pt", map_location="cuda"))
mr,_ = load_maskrcnn(42); rng = np.random.default_rng(0); tiles=[]
from pycocotools import mask as mu
for cid, img, valid, gt in crops("validation"):
    dets, f0 = maskrcnn_forward(mr, np.ascontiguousarray(img))
    cand=[(d[0]&valid,d[1]) for d in dets if (d[0]&valid).any()]; ms=[c[0] for c in cand]
    labs=candidate_labels(gt, ms); ridx=[j for j,(k,_) in enumerate(labs) if k>=2]
    if not ridx: continue
    boxes=[roi_box(ms[j]) for j in range(len(ms))]
    F=roi_features(f0,boxes).cuda().float(); M=torch.from_numpy(np.stack([crop_resample(x.astype(np.float32),b) for x,b in zip(ms,boxes)])[:,None]).cuda()
    with torch.no_grad(): o={k:v.cpu().numpy() for k,v in m(F,M).items()}
    H,W=gt.shape
    for j in ridx:
        if rng.random()>0.12 or len(tiles)>=8: continue
        kids=children_M2(o["member_logits"][j], boxes[j], H, W, valid); kids=[k for k in kids if k.sum()>=100]
        decided = o["logit"][j]+shift>0
        rgb=cv2.cvtColor(cv2.normalize(img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8),cv2.COLOR_RGB2BGR).copy()
        x0,y0,x1,y1=boxes[j]; cx,cy=(x0+x1)//2,(y0+y1)//2; s=max(x1-x0,y1-y0,80)
        yy0,xx0=np.clip(cy-s,0,H-2*s),np.clip(cx-s,0,W-2*s) if 2*s<W else 0
        def ol(mask,col,th=2):
            cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(rgb,cs,-1,col,th)
        for v in np.unique(gt)[1:]: ol(gt==v,(0,220,220),1)
        for kk,col in zip(kids,[(255,255,0),(255,0,255)]): ol(kk,col,2)
        ol(ms[j],(0,0,255),2)
        crop=cv2.resize(rgb[yy0:yy0+2*s, (xx0 if isinstance(xx0,int) else int(xx0)):(xx0 if isinstance(xx0,int) else int(xx0))+2*s],(320,320))
        cv2.putText(crop,f"{cid} {'REFORM' if decided else 'KEEP'} k{labs[j][0]}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        tiles.append(crop)
    if len(tiles)>=8: break
while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
grid=np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])
p=STUDY/"previews"/"m2_reform_examples.png"; p.parent.mkdir(exist_ok=True); cv2.imwrite(str(p),grid); print(p)
