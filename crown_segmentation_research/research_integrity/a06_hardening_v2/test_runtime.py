"""Regression on the real loader with persistent workers; no model training."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from crown_segmentation_research.clean_benchmark import dataset as production

class ProductionLoaderTests(unittest.TestCase):
    def test_shared_planned_window_drives_real_field_and_instance_loaders(self):
        from crown_segmentation_research.clean_benchmark import train_maskrcnn as mr
        from .core import encode,IntegrityError
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'train').mkdir();yy,xx=np.indices((20,20));img=np.stack([yy*10,xx*10,xx+yy],axis=-1).astype(np.uint8)
            large=np.zeros((20,20),bool);large[2:18,2:18]=1;tiny=np.zeros_like(large);tiny[8,8]=1
            np.savez(root/'train/t.npz',image=img,valid=np.ones_like(large),label=large.astype(np.int32))
            anns=[]
            for i,m in enumerate((large,tiny),1):
                encoded=encode(m);encoded['counts']=encoded['counts'].decode();anns.append({'id':i,'image_id':1,'segmentation':encoded,'area':int(m.sum())})
            (root/'train_coco.json').write_text(json.dumps({'images':[{'id':1,'file_name':'t.tif','height':20,'width':20,'bounds':[0,0,.6,.6]}],'annotations':anns}))
            plan=[{'tile_name':'t.npz','window_yxhw':[5,5,10,10],'annotation_ids':[1,2],'visible_target_count':2,'categories':['partial','one_complete']}]
            oldf,oldm=production.PREP,mr.PREP;production.PREP=mr.PREP=root
            try:
                f=production.CanopyCropDataset('train',crop=10,augment=False,master_seed=7,sample_plan=plan)
                q=mr.CanopyInstanceCrops('train',crop=10,augment=False,master_seed=7,sample_plan=plan)
                batch=f[0];image,target=q[0]
                recovered=batch['image'].numpy()*production.STD[:,None,None]+production.MEAN[:,None,None]
                self.assertTrue(np.allclose(recovered,image.numpy(),atol=1e-6));self.assertEqual(len(target['masks']),2)
                self.assertEqual(json.loads(batch['integrity_record'])['field_invisible_annotation_ids'],[2])
                plan[0]['annotation_ids']=[1]
                with self.assertRaises(IntegrityError):q[0]
            finally:production.PREP,mr.PREP=oldf,oldm

    def test_native_RLE_crop_mask_and_area_match_decode(self):
        from .core import encode
        from .sampling import crop_mask,crop_area
        rng=np.random.default_rng(20261015)
        for shape in ((1,1),(7,9),(31,25)):
            for _ in range(12):
                m=rng.random(shape)>.5;r=encode(m)
                y=int(rng.integers(shape[0]));x=int(rng.integers(shape[1]));h=int(rng.integers(1,shape[0]-y+1));w=int(rng.integers(1,shape[1]-x+1))
                window=(y,x,h,w);expected=m[y:y+h,x:x+w]
                self.assertTrue(np.array_equal(crop_mask(r,window),expected))
                self.assertEqual(crop_area(r,window),int(expected.sum()))

    def test_field_validation_epoch_locked_and_workers_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'validation').mkdir();yy,xx=np.indices((40,40));image=np.stack([yy*6,xx*6,xx+yy],axis=-1).astype(np.uint8)
            np.savez(root/'validation/t.npz',image=image,valid=np.ones((40,40),bool),label=np.ones((40,40),np.int32))
            old=production.PREP;production.PREP=root
            try:
                ds=production.CanopyCropDataset('validation',crop=12,crops_per_tile=1,fixed=True,augment=False)
                ds.set_epoch(0);a=ds[0]['image'];ds.set_epoch(99);b=ds[0]['image'];self.assertTrue(torch.equal(a,b))
                dl=DataLoader(ds,batch_size=1,num_workers=2)
                self.assertTrue(torch.equal(a,next(iter(dl))['image'][0]))
            finally:production.PREP=old

    def test_instance_loader_keeps_nested_one_pixel_original_target(self):
        from crown_segmentation_research.clean_benchmark import train_maskrcnn as mr
        from .core import encode
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'train').mkdir()
            large=np.zeros((20,20),bool);large[2:18,2:18]=1
            tiny=np.zeros_like(large);tiny[8,8]=1
            np.savez_compressed(root/'train/t.npz',image=np.zeros((20,20,3),np.uint8),valid=np.ones_like(large),label=large.astype(np.int32))
            anns=[]
            for i,m in enumerate((large,tiny),1):
                r=encode(m);r['counts']=r['counts'].decode();anns.append({'id':i,'image_id':1,'segmentation':r,'area':int(m.sum()),'bbox':[0,0,20,20]})
            (root/'train_coco.json').write_text(json.dumps({'images':[{'id':1,'file_name':'t.tif','height':20,'width':20,'bounds':[0,0,.6,.6],'crs':'EPSG:32618'}],'annotations':anns}))
            old=mr.PREP;mr.PREP=root
            try:
                ds=mr.CanopyInstanceCrops('train',crop=20,augment=False,fixed=True)
                _,target=ds[0]
                self.assertEqual(len(target['masks']),2,'Overlapping one-pixel original GT must not disappear via priority raster/min area')
                self.assertEqual(target['annotation_ids'].tolist(),[1,2])
            finally:mr.PREP=old

    def test_persistent_worker_epoch_changes_real_crops(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'train').mkdir();yy,xx=np.indices((40,40))
            image=np.stack([yy*6,xx*6,(xx+yy)*3],axis=-1).astype(np.uint8)
            for i in range(4):np.savez_compressed(root/'train'/f'{i}.npz',image=image,valid=np.ones((40,40),bool),label=np.ones((40,40),np.int32))
            old=production.PREP;production.PREP=root
            try:
                ds=production.CanopyCropDataset('train',crop=12,crops_per_tile=1,augment=False)
                self.assertTrue(hasattr(ds,'set_epoch'),'Production loader must expose epoch API, not helper-only fix')
                dl=DataLoader(ds,batch_size=1,num_workers=2,persistent_workers=True)
                try:
                    ds.set_epoch(0);a=[b['image'].clone() for b in dl]
                    ds.set_epoch(1);b=[b['image'].clone() for b in dl]
                    self.assertTrue(any(not torch.equal(x,y) for x,y in zip(a,b)))
                    ds.set_epoch(0);c=[b['image'].clone() for b in dl]
                    self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,c)))
                finally:
                    if dl._iterator is not None:dl._iterator._shutdown_workers()
            finally:production.PREP=old

if __name__=='__main__':unittest.main()
