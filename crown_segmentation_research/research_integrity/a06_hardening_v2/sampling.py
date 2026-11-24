"""Worker-shared epoch state and native-RLE crop accounting for production loaders."""
import collections
import json
from pathlib import Path
import numpy as np
import torch
from pycocotools import mask as mu
from .core import REPO,affine_from_bounds,pixel_area,require,rle,rng_for,source_key

class EpochSampling:
    def init_sampling(self,split,master_seed=None,epoch_rng=True,sample_plan=None):
        self.split=split;self.master_seed=int(torch.initial_seed() if master_seed is None else master_seed)
        self.epoch_rng=epoch_rng;self._epoch_shared=torch.zeros((),dtype=torch.int64).share_memory_()
        self.sample_plan=sample_plan
        if sample_plan is not None:
            require(isinstance(sample_plan,list) and bool(sample_plan),"Nonempty frozen crop plan required")
            files={p.name for p in self.files}
            require(all(r.get('tile_name') in files and len(r.get('window_yxhw',[]))==4 for r in sample_plan),"Crop plan source/window missing or outside dataset")
    def sample_path(self,index):
        return self.files[index//self.cpt] if self.sample_plan is None else next(p for p in self.files if p.name==self.sample_plan[index]['tile_name'])
    def sample_length(self):return len(self.files)*self.cpt if self.sample_plan is None else len(self.sample_plan)
    def planned_window(self,index):return None if self.sample_plan is None else self.sample_plan[index]['window_yxhw']
    def validate_planned_record(self,index,record):
        if self.sample_plan is None:return
        declared=self.sample_plan[index]
        require(sorted(record.get('annotation_ids',[]))==sorted(declared.get('annotation_ids',[])),"Crop plan GT identity changed; no silent target loss")
        require(record['visible_target_count']==declared.get('visible_target_count'),"Crop plan target count changed")
        require(set(declared.get('categories',[]))<=set(record['categories']),"Crop plan category not supported by actual native masks")
    def set_epoch(self,epoch):
        require(isinstance(epoch,int) and epoch>=0,"Nonnegative epoch required")
        self._epoch_shared.fill_(0 if self.fixed else epoch)
    def sample_rng(self,index):
        if self.fixed:return np.random.default_rng(index) # preserve historical validation crop locations
        if not self.epoch_rng:return np.random.default_rng(index if self.fixed else (torch.initial_seed()+index)%(2**32))
        return rng_for(self.master_seed,int(self._epoch_shared.item()),self.split,self.sample_path(index).name,index if self.sample_plan is not None else index%self.cpt,self.fixed)

def transformed_crop_affine(affine,window,flip_x,flip_y,k):
    y,x,h,w=window;a,b,c,d,e,f=affine
    t=np.array([[a,b,c+a*x+b*y],[d,e,f+d*x+e*y],[0,0,1]],float)
    if flip_x:t=t@np.array([[-1,0,w],[0,1,0],[0,0,1]])
    if flip_y:t=t@np.array([[1,0,0],[0,-1,h],[0,0,1]])
    rotations=(np.eye(3),np.array([[0,-1,w],[1,0,0],[0,0,1]]),np.array([[-1,0,w],[0,-1,h],[0,0,1]]),np.array([[0,1,0],[-1,0,h],[0,0,1]]))
    t=t@rotations[k%4];return [float(v) for v in t[:2].ravel()]

def annotation_index(root,split,reference_metadata=None):
    path=Path(root)/f'{split}_coco.json'
    if not path.exists():return {} # synthetic/in-memory callers; runtime gate requires source COCO.
    coco=json.loads(path.read_text());by=collections.defaultdict(list)
    for ann in coco['annotations']:by[ann['image_id']].append(ann)
    index={}
    for im in coco['images']:
        im=dict(im);npz=(Path(root)/split/(Path(im['file_name']).stem+'.npz')).resolve()
        if reference_metadata is not None:
            meta=reference_metadata.get(source_key(npz))
            require(meta is not None,"Actual affine reference missing for training tile")
            im['affine']=meta['affine'];im['affine_evidence']=meta['evidence']
        index[Path(im['file_name']).stem]=(im,by[im['id']])
    return index

def crop_area(r,window):
    """Count foreground∩rectangle by cumulative column-major rectangle support, without decoding full image."""
    rr=rle(r);H,W=rr['size'];y,x,h,w=window
    require(min(y,x)>=0 and y+h<=H and x+w<=W,"Window outside RLE")
    import importlib
    counts=importlib.import_module('crown_segmentation_research.diagnosis_2026-10-11.fgq_revision_2026-10-13.grouping').rle_counts(r)
    ends=np.cumsum(counts,dtype=np.int64);starts=np.r_[0,ends[:-1]]
    def prefix(n):
        col,row=n//H,n%H
        return np.clip(col-x,0,w)*h+np.where((col>=x)&(col<x+w),np.clip(row-y,0,h),0)
    return int((prefix(ends[1::2])-prefix(starts[1::2])).sum())

def crop_mask(r,window):
    rr=rle(r);H,W=rr['size'];y,x,h,w=window
    require(min(y,x)>=0 and y+h<=H and x+w<=W,"Window outside RLE")
    import importlib
    counts=importlib.import_module('crown_segmentation_research.diagnosis_2026-10-11.fgq_revision_2026-10-13.grouping').rle_counts(r)
    ends=np.cumsum(counts,dtype=np.int64);starts=np.r_[0,ends[:-1]];out=np.zeros((h,w),bool)
    for s,e in zip(starts[1::2],ends[1::2]):
        if e<=s:continue
        for col in range(max(x,int(s//H)),min(x+w,int((e-1)//H)+1)):
            ya=max(y,int(s-col*H));yb=min(y+h,int(e-col*H))
            if yb>ya:out[ya-y:yb-y,col-x]=True
    return out

def describe_sample(index,path,label,valid,window,shape_tags=False):
    y,x,h,w=window;entry=index.get(Path(path).stem)
    if entry is None:
        return {'source':str(path),'window_yxhw':list(window),'source_reference':'synthetic priority raster only; original annotation provenance UNKNOWN',
                'complete_count':None,'partial_count':None,'visible_target_count':len(np.unique(label[y:y+h,x:x+w][label[y:y+h,x:x+w]>0])),
                'valid_fraction':float(valid[y:y+h,x:x+w].mean()),'physical_crop_area_m2':None}
    im,anns=entry;affine=im.get('affine',affine_from_bounds(im['bounds'],valid.shape));px=pixel_area(affine)
    complete=[];partial=[];visible=[];missing=[];small=[];ambiguous=[]
    priority_ids=set(np.unique(label[y:y+h,x:x+w]).tolist())
    for j,ann in enumerate(anns,1):
        n=crop_area(ann['segmentation'],window)
        if not n:continue
        aid=ann['id'];visible.append(aid)
        full=int(mu.area(rle(ann['segmentation'])))
        bx,by,bw,bh=mu.toBbox(rle(ann['segmentation']))
        edge=bx<=0 or by<=0 or bx+bw>=valid.shape[1] or by+bh>=valid.shape[0]
        (ambiguous if n==full and edge else complete if n==full else partial).append(aid)
        if full*px<1:small.append(aid)
        if j not in priority_ids:missing.append(aid)
    categories=[]
    if not visible:categories.append('background')
    if len(complete)==1:categories.append('one_complete')
    if 2<=len(complete)<=4:categories.append('two_to_four_complete')
    if len(visible)>4:categories.append('crowded')
    if partial:categories.append('partial')
    if small:categories.append('small')
    if shape_tags:
        from scipy import ndimage as ndi
        ms=[crop_mask(a['segmentation'],window) for a in anns if a['id'] in visible]
        if any(ndi.label((lambda d:(d==ndi.maximum_filter(d,size=3))&(d>1))(ndi.distance_transform_edt(m)))[1]>1 for m in ms):categories.append('multilobed')
        if any((ndi.binary_dilation(a)&b).any() for i,a in enumerate(ms) for b in ms[:i]):categories.append('touching')
    return {'source':str(path),'image_id':im['id'],'annotation_ids':visible,'complete_annotation_ids':complete,'partial_annotation_ids':partial,
            'complete_count':len(complete),'partial_count':len(partial),'visible_fragments':len(partial),'ignored_count':0,
            'ignored_annotation_ids':[],'ambiguous_target_count':len(ambiguous),'ambiguous_annotation_ids':ambiguous,'query_capacity_requirement':len(visible),'visible_target_count':len(visible),
            'priority_missing_annotation_ids':missing,'reference_semantics':'original RLE counted; priority ownership remains a separate field supervision',
            'window_yxhw':list(window),'valid_fraction':float(valid[y:y+h,x:x+w].mean()),'affine':affine,
            'physical_crop_area_m2':h*w*px,'GSD_axes_m':[abs(affine[0]),abs(affine[4])],'categories':categories,
            'affine_evidence':im.get('affine_evidence','bounds reconstruction; synthetic/library path only, runtime requires actual transform'),
            'shape_tags':'native-mask proxies evaluated' if shape_tags else 'touching/multilobed not assessed in random sampler metadata',
            'partial_policy':'visible_fragment_retains_original_identity'}

def validate_frozen_plan(plan,prepared_root,metadata,max_targets,min_valid,queries=None):
    require(plan.get('status')=='PASS' and plan.get('records'),"Frozen crop plan not PASS")
    quotas=plan.get('category_quotas');require(isinstance(quotas,dict) and set(quotas)=={'background','one_complete','two_to_four_complete','crowded','small','multilobed','touching','partial'},"Explicit category quotas required")
    require(all(isinstance(n,int) and n>=0 for n in quotas.values()),"Invalid category quota")
    indices={split:annotation_index(prepared_root,split,metadata) for split in ('train','validation')}
    counts={k:0 for k in quotas};maximum=0
    for declared in plan['records']:
        split=declared['split'];require(split in indices,"Crop plan cannot sample historical test labels")
        path=Path(prepared_root)/split/declared['tile_name'];require(path.name==declared['tile_name'],"Crop tile path traversal forbidden")
        with np.load(path) as z:label=z['label'];valid=z['valid']
        y,x,h,w=declared['window_yxhw'];require(min(y,x)>=0 and min(h,w)>0 and y+h<=label.shape[0] and x+w<=label.shape[1],"Crop plan outside source")
        record=describe_sample(indices[split],path,label,valid,(y,x,h,w),shape_tags=True)
        require(record['valid_fraction']>=min_valid,"Frozen crop violates support policy")
        require(record['annotation_ids']==declared['annotation_ids'] and record['visible_target_count']==declared['visible_target_count'],"Frozen plan GT targets do not match original RLE")
        require(set(declared['categories'])<=set(record['categories']),"Frozen plan category does not match data")
        require(record['visible_target_count']<=max_targets,"Actual plan GT count exceeds declared capacity")
        if queries is not None:
            from .core import check_capacity
            check_capacity(record['visible_target_count'],queries)
        maximum=max(maximum,record['visible_target_count'])
        if split=='train':
            for k in record['categories']:counts[k]+=1
    require(all(counts[k]>=n for k,n in quotas.items()),"Actual native-mask category quotas unmet; do not hide difficult strata")
    return {'validated_records':len(plan['records']),'train_category_counts':counts,'maximum_target_count':maximum}
