"""Fixed metadata-selected source/reference/frozen-prediction illustrations."""
from pathlib import Path
import sys,json,gzip,hashlib,csv
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pycocotools import mask as mu
sys.path.insert(0,'/home/chung/RS/data/TreeCrown-MultiSource-V1')
from treecrown_data import TreeCrownData
from treecrown_data.masks import unpack
EXP=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def rgzip(p):
    with gzip.open(p,'rt') as f:return json.load(f)
def outlines(masks,shape):
    edge=np.zeros(shape,bool)
    for r in masks:
        m=mu.decode(unpack(r)).astype(bool)
        edge|=m & ~ndi.binary_erosion(m)
    return ndi.binary_dilation(edge,iterations=2)
def main():
    assert (EXP/'results/COMPLETE.json').is_file()
    c=read(EXP/'metadata/SELECTED_COHORT.json')['selected']
    groups=sorted({r['spatial_group'] for r in c})
    chosen=[next(i for i,r in enumerate(c) if r['spatial_group']==g) for g in groups]
    dst=EXP/'figures';dst.mkdir(exist_ok=True)
    D=TreeCrownData.from_root('/home/chung/RS/data/TreeCrown-MultiSource-V1')
    reference=read(EXP/'results/REFERENCE_METADATA.json')['references']
    columns=[('RGB',None),('Published references','reference'),('R5','r5_seed42'),('Mask2Former','m2f_seed42'),('DINO original','dino_seed42'),('Matched continuation','control_seed42'),('RR_MASK_INIT','rr_seed42')]
    manifest={'selection_rule':'first metadata-selected tile per geographic component; no outcome selection','tiles':[],'overlays':'individual raster mask outlines; no exclusive ownership projection','image_adjustment':'none; invalid pixels grey only for display; nearest display sampling','scale':'per-tile GeoTIFF spacing;10m horizontal bar','independent_identity_verification':False,'publisher_requirements':'provisional general research figure; no journal compliance claim'}
    with plt.rc_context({'font.size':9,'axes.titlesize':10}):
        fig,axes=plt.subplots(len(chosen),len(columns),figsize=(21,6.9),layout='constrained',squeeze=False)
        for rownum,i in enumerate(chosen):
            r=c[i];s=D.sample_by_id(r['image_id'],role='development');rgb=s['rgb'].copy();v=s['valid_mask'];rgb[~v]=170
            valid=mu.encode(np.asfortranarray(v.astype(np.uint8)))
            gt=[]
            for rr in s['instance_masks'].rles:
                clipped=mu.merge([unpack(rr),valid],intersect=True)
                if mu.area(clipped):gt.append(clipped)
            for col,(title,arm) in enumerate(columns):
                image=rgb.copy();n=''
                if arm=='reference':
                    edge=outlines(gt,v.shape);image[edge]=[0,255,255];n=f' ({len(gt)})'
                elif arm is not None:
                    p=rgzip(EXP/'predictions'/arm/f'b{i:03d}.json.gz');ps=[p['raw_predictions'][j]['rle'] for j in p['final_indices']]
                    edge=outlines(ps,v.shape);image[edge]=[255,140,0];n=f' ({len(ps)})'
                ax=axes[rownum,col];ax.imshow(image,interpolation='nearest');ax.set_title(title+n);ax.set_xticks([]);ax.set_yticks([])
                ax.plot([90,90+10/r['gx_m']],[1960,1960],color='white',lw=2,path_effects=[pe.Stroke(linewidth=4,foreground='black'),pe.Normal()])
                ax.text(90,1920,'10 m',color='white',fontsize=8,path_effects=[pe.Stroke(linewidth=2,foreground='black'),pe.Normal()])
                if col==0:ax.set_ylabel(f"{r['spatial_group']}\nGSD {r['gx_m']*100:.3f} cm/px\n{i:03d}")
            manifest['tiles'].append(dict(cohort_index=i,image_id=r['image_id'],spatial_group=r['spatial_group'],RGB_sha256=hashlib.sha256(s['rgb'].tobytes()).hexdigest(),reference_count=len(gt),reference_metadata_count=len(reference[i])))
        fig.suptitle('Frozen carriers on native BAM TRAIN imagery — fixed examples, published-reference fidelity',fontsize=13)
        fig.supxlabel('Cyan: reference outlines. Orange: accepted independent prediction outlines. Counts are per tile; original crown area and identity reliability remain unverified.',fontsize=10)
        fig.savefig(dst/'FIXED_CROSS_CARRIER_COMPARISON.png',dpi=200)
        plt.close(fig)
    with (EXP/'results/RR_PAIRED_RESULTS.csv').open() as f:paired=list(csv.DictReader(f))
    seeds=[int(r['seed']) for r in paired];x=np.arange(len(seeds));new=np.array([int(r['new']) for r in paired]);lost=np.array([int(r['lost']) for r in paired])
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    ax.bar(x-.18,new,width=.36,color='#0072B2',label='New reference matches')
    ax.bar(x+.18,-lost,width=.36,color='#D55E00',hatch='//',label='Lost reference matches')
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,[str(s) for s in seeds]);ax.set_xlabel('Paired computational seed (same24tiles, not independent geography)');ax.set_ylabel('Reference occurrences gained (+) / lost (−)');ax.set_title('RR versus matched continuation: gross gains and losses');ax.legend()
    for j in range(len(seeds)):
        ax.text(x[j]-.18,new[j]+1,str(new[j]),ha='center',va='bottom');ax.text(x[j]+.18,-lost[j]-1,str(lost[j]),ha='center',va='top')
    low,high=ax.get_ylim();ax.set_ylim(low-min(3,(high-low)*.07),high+min(3,(high-low)*.07))
    fig.savefig(dst/'RR_NEW_AND_LOST.png',dpi=180);plt.close(fig)
    manifest['alt_text']='Two fixed geographic-component examples show the same native RGB and reference/five carrier outlines; no identity validation is implied. A separate diverging bar plot displays each paired seed’s new and lost matches, including losses.'
    (dst/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'fixed_tiles':chosen,'figures':2,'reference_stability':'UNVERIFIED'}))
if __name__=='__main__':main()
