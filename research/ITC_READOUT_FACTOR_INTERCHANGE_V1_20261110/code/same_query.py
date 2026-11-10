from common import *
import numpy as np
from pycocotools import mask as mu
def rle(r):return dict(size=r['size'],counts=r['counts'].encode())
def main():
    assert (EXP/'VERIFICATION.json').exists()
    gt=gz(CONTROL/'evaluations/S0/results/REFERENCE_MASKS.json.gz');cohort=read(PILOT/'COHORT.json')
    subset=rows(EXP/'results/S1_STRONG_RESIDUAL_SUBSET.csv');records=[]
    for z in subset:
        i,j=next((i,j) for i,mm in enumerate(gt['metadata']) for j,r in enumerate(mm) if r['reference_id']==z['reference_id'])
        y=rle(gt['references'][i][j]['rle']);gx,gy,gw,gh=mu.toBbox(y)/1024;gb=np.array([gx,gy,gx+gw,gy+gh])
        for code in ['AB','BA']:
            d=gz(EXP/f"predictions/S1/{cohort[i]['id']}_{code}.json.gz");native=gz(EXP/f"predictions/S1/{cohort[i]['id']}_{code[0]*2}.json.gz")
            a=mu.iou([rle(r['rle']) for r in d['raw_predictions']],[y],[0])[:,0];q=int(a.argmax());p=d['raw_predictions'][q];p0=native['raw_predictions'][q]
            for key in ['encoder_site','class_logit','class_probability','predicted_box_cxcywh']:assert p[key]==p0[key]
            iou0=float(mu.iou([rle(p0['rle'])],[y],[0])[0,0]);cx,cy,w,h=p['predicted_box_cxcywh'];b=np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
            inter=np.maximum(np.minimum(b[2:],gb[2:])-np.maximum(b[:2],gb[:2]),0).prod();boxiou=inter/(w*h+gw*gh-inter)
            records.append(dict(reference_id=z['reference_id'],source=z['source'],source_small=z['source_small'],cross_readout=code,
                    source_query_id=q,source_encoder_site=p['encoder_site'],source_class_probability=p['class_probability'],
                    source_query_box_iou=float(boxiou),same_query_factual_mask_iou=iou0,cross_mask_iou=float(a[q]),
                    qualifies_after_swap=bool(a[q]>=.5),final_retained_after_swap=q in d['final_indices'],
                    GT_conditioned_witness_selection=True,GT_inference=False))
    assert len(records)==40;csvwrite(EXP/'results/EXPLORATORY_SAME_QUERY.csv',records)
    summary=[]
    for code in ['AB','BA']:
        part=[r for r in records if r['cross_readout']==code and r['qualifies_after_swap']]
        summary.append(dict(readout=code,qualifying_witnesses=len(part),same_query_box_iou50=sum(r['source_query_box_iou']>=.5 for r in part),
                      same_query_factual_mask_iou50=sum(r['same_query_factual_mask_iou']>=.5 for r in part),retained=sum(r['final_retained_after_swap'] for r in part)))
    write(EXP/'results/EXPLORATORY_SAME_QUERY_SUMMARY.json',dict(exploratory=True,summary=summary,thresholds_changed=False,forwards_added=0))
    print(json.dumps(summary))
if __name__=='__main__':main()
