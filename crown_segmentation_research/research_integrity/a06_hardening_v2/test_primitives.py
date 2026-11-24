"""Fast production-API regression suite executed by preflight (no model/data files)."""
import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
from .core import (IntegrityError,affine_from_bounds,area,binary,check_access,check_capacity,encode,pixel_area,resized_affine,rle,rng_for,sha256,size_bin,supports,verify_manifest,write_new)
from .crops import choose_window,crown_centered_window,record_crop,synchronized_transform
from .metrics import (boundary_f1,candidate_strata,coco_ap,conditioned_recall,core_mask,count_statistics,grouping,iou_matrix,match_matrix,overlap_diagnostics,recall)
from .outputs import Instance,accept,materialize_q,stitch,validate
from .spatial import validate_split

def mask(y=2,x=2,h=3,w=3,shape=(12,12)):
    m=np.zeros(shape,bool);m[y:y+h,x:x+w]=1;return m

class ReferenceTests(unittest.TestCase):
    def test_six_supports_and_original_preserved(self):
        a=mask(1,1,8,8);b=mask(3,3,2,2);v=np.ones_like(a);v[0:4]=0
        s=supports([a,b],v,[100,200]);self.assertEqual(s['unique_original'][1].sum(),0)
        self.assertEqual(s['priority'][1].sum(),0);self.assertTrue(np.array_equal(s['original'][1],b))
        self.assertEqual(s['valid'][0].sum(),40);self.assertFalse(s['priority_valid'][0][~v].any())
    def test_stable_owner_order_invariant_and_legacy_explicit(self):
        a=mask(1,1,8,8);b=mask(3,3,2,2);v=np.ones_like(a)
        s=supports([a,b],v,[2,1],'stable_id_first_wins');t=supports([b,a],v,[1,2],'stable_id_first_wins')
        self.assertTrue(np.array_equal(s['priority'][1],t['priority'][0]))
        self.assertFalse(np.array_equal(supports([a,b],v,[2,1])['priority'][0],s['priority'][0]))
    def test_bad_shape_and_values(self):
        for m in (np.ones((12,1)),np.full((12,12),np.nan),np.full((12,12),.5)):
            with self.assertRaises(IntegrityError):binary(m,(12,12))
    def test_RLE_roundtrip_and_reencoding(self):
        rng=np.random.default_rng(8)
        for shape in ((1,1),(7,9),(31,25)):
            m=rng.random(shape)>.7;r=encode(m)
            self.assertTrue(np.array_equal(mu.decode(rle(r,shape)).astype(bool),m))
            self.assertTrue(np.array_equal(mu.decode(encode(mu.decode(r))),m))
    def test_RLE_invalid_runs_dimensions(self):
        for r in ({'size':[2,2],'counts':[0,5]},{'size':[2,2],'counts':[0,-1,5]},{'size':[0,2],'counts':[0]}, {'size':[2,2],'counts':[1.2,2.8]}, {'size':[2,2],'counts':b'\x00'}):
            with self.assertRaises(IntegrityError):rle(r)
        with self.assertRaises(IntegrityError):rle(encode(mask()),(3,4))
    def test_affine_area_nonsquare_rotated_and_resize(self):
        m=mask(h=2,w=4);a=(.04,.01,100,.02,-.05,200)
        self.assertAlmostEqual(area(encode(m),a),8*.0022)
        scaled=resized_affine(a,(12,12),(24,36));self.assertAlmostEqual(pixel_area(scaled)*6,pixel_area(a))
    def test_area_bin_boundaries_and_invalid(self):
        self.assertEqual([size_bin(a) for a in (0,.9999,1,2,4,9)],['lt1','lt1','1to2','2to4','4to9','ge9'])
        for a in (float('nan'),float('inf'),-1):
            with self.assertRaises(IntegrityError):size_bin(a)
        with self.assertRaises(IntegrityError):pixel_area([0]*6)
    def test_metric_CRS_rejects_degree_and_feet_axes(self):
        from .core import metric_crs
        self.assertEqual(metric_crs('EPSG:32618'),'EPSG:32618')
        for crs in ('EPSG:4326','EPSG:2263'):
            with self.assertRaises(IntegrityError):metric_crs(crs)
    def test_hash_missing_empty_changed_and_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x';p.write_text('abc');manifest={'x':{'sha256':sha256(p)}}
            self.assertEqual(len(verify_manifest(manifest,td,['x'])),1)
            for bad in ({},{'x':'MISSING'},{'x':{'sha256':'0'*64}},{'../x':{'sha256':'0'*64}}):
                with self.assertRaises(IntegrityError):verify_manifest(bad,td,['x'])
    def test_exclusive_artifact_destination(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'artifact.json';write_new(p,{'x':1})
            with self.assertRaises(FileExistsError):write_new(p,{'x':2})
            self.assertIn('1',p.read_text())
    def test_roles_reject_test_selection_training_calibration(self):
        for purpose in ('training','preprocessing_fit','model_selection','threshold_selection','calibration'):
            with self.assertRaises(IntegrityError):check_access('CONFIRMATORY_TEST',purpose)
        check_access('TRAIN','training');check_access('LEGACY_TEST','integrity_scan')

class MetricTests(unittest.TestCase):
    def test_perfect_empty_duplicate_merge_split(self):
        a,b=mask(),mask(7,7)
        for gt,pred,n in (([a,b],[a,b],2),([a],[],0),([],[a],0),([a],[a,a,a],1),([a,b],[a|b],1)):
            self.assertEqual(len(match_matrix(iou_matrix(gt,pred))),n)
        g=mask(1,1,6,6);p=mask(1,1,3,6);q=mask(4,1,3,6)
        self.assertEqual(len(match_matrix(iou_matrix([g],[p,q]))),1)
    def test_cardinality_beats_largest_sum_greedy(self):
        m=np.array([[.99,.5],[.5,.49]])
        self.assertEqual(len(match_matrix(m)),2)
    def test_cardinality_matches_bruteforce(self):
        rng=np.random.default_rng(17)
        for _ in range(12):
            a=rng.random((4,4));best=(-1,-1.)
            for perm in itertools.permutations(range(4)):
                vals=[a[i,j] for i,j in enumerate(perm) if a[i,j]>=.5];best=max(best,(len(vals),sum(vals)))
            found=match_matrix(a);self.assertEqual(len(found),best[0]);self.assertAlmostEqual(sum(v for i,j,v in found),best[1])
    def test_overlapping_GT_not_owner_raster(self):
        a=mask(1,1,8,8);b=mask(3,3,2,2)
        self.assertEqual(recall([a,b],[a,b],[4,.1])['tp'],2)
    def test_area_strata_joint_matching_and_NA(self):
        a,b=mask(),mask(7,7);r=recall([a,b],[a],[.1,2.])
        self.assertEqual(r['bins']['lt1']['recall'],1);self.assertEqual(r['bins']['2to4']['recall'],0)
        self.assertIsNone(r['bins']['ge9']['recall']);self.assertIsNone(recall([],[],[])['recall'])
    def test_grouping_perfect_abstain_all_merge_split(self):
        pairs=np.array([[0,1],[0,2]]);same=np.array([True,False])
        perfect=np.array([[1,1,0],[0,0,1]],bool)
        self.assertEqual(grouping(perfect,pairs,same)['balanced'],1)
        self.assertEqual(grouping(np.ones((1,3),bool),pairs,same)['different']['unconditional'],0)
        self.assertEqual(grouping(np.eye(3,dtype=bool),pairs,same)['same']['unconditional'],0)
        self.assertEqual(grouping(np.vstack([perfect,perfect]),pairs,same)['pair_coverage'],0)
    def test_grouping_NA_counts_and_coverage(self):
        r=grouping(np.zeros((0,3),bool),[[0,1]],[True])
        self.assertEqual(r['same']['unconditional'],0);self.assertIsNone(r['same']['conditional'])
        self.assertIsNone(r['different']['unconditional']);self.assertIsNone(r['balanced'])
        r=grouping(np.zeros((0,0),bool),[],[]);self.assertIsNone(r['pair_coverage']);self.assertEqual(r['n_pairs'],0)
    def test_score_only_structural_invariant_E2E_changes(self):
        cover=np.array([[1,1,0],[1,0,1]],bool);p=[[0,1]];s=[True]
        a=grouping(cover,p,s,[.9,.8]);b=grouping(cover,p,s,[.8,.9]);self.assertEqual(a,b)
        self.assertEqual(grouping(cover,p,s,[.9,.8],.5)['same']['unconditional'],1)
        self.assertEqual(grouping(cover,p,s,[.8,.9],.5)['same']['unconditional'],0)
    def test_structural_reorder_and_no_score_cap(self):
        c=np.array([[1,1,0],[0,0,1]],bool)
        self.assertEqual(grouping(c,[[0,1],[0,2]],[True,False]),grouping(c[::-1],[[0,1],[0,2]],[True,False]))
        with self.assertRaises(IntegrityError):grouping(c,[],[],max_dets=400)
    def test_COCO_perfect_empty_and_score_ranking(self):
        a=mask();false=mask(7,7)
        self.assertAlmostEqual(coco_ap([a],[a],[.9])['AP'],1)
        self.assertEqual(coco_ap([a],[],[])['AP'],0);self.assertIsNone(coco_ap([],[a],[.9])['AP'])
        self.assertGreater(coco_ap([a],[a,false],[.9,.1])['AP'],coco_ap([a],[a,false],[.1,.9])['AP'])
    def test_COCO_mask_ID_permutation_is_irrelevant(self):
        a,b=mask(),mask(7,7)
        self.assertEqual(coco_ap([a,b],[a,b],[.9,.8]),coco_ap([b,a],[b,a],[.8,.9]))
    def test_physical_AP_bins_and_nonsquare_pixels(self):
        a=mask(h=2,w=4);affine=(.5,0,0,0,-.25,0) # 8px=1m2 exactly
        self.assertAlmostEqual(coco_ap([a],[a],[.9],affine=affine,area_range_m2=(1,2))['AP'],1)
        self.assertIsNone(coco_ap([a],[a],[.9],affine=affine,area_range_m2=(0,1))['AP'])
    def test_E2E_ties_do_not_depend_on_renamed_instance_IDs(self):
        c=np.array([[1,1,0],[1,0,1]],bool)
        a=grouping(c,[[0,1]],[True],[.8,.8],.5,stable_ids=['a','b'])
        b=grouping(c,[[0,1]],[True],[.8,.8],.5,stable_ids=['b','a'])
        self.assertEqual(a,b)
    def test_boundary_small_border_tolerance(self):
        a=mask(0,0,3,3);b=np.roll(a,1,axis=1)
        self.assertEqual(boundary_f1(a,a,0)['F1'],1)
        self.assertLess(boundary_f1(b,a,0)['F1'],boundary_f1(b,a,1)['F1'])
        self.assertEqual(boundary_f1(np.zeros_like(a),a,1)['F1'],0)
    def test_boundary_metric_uses_ground_affine(self):
        a=mask();b=np.roll(a,1,axis=1);affine=(.04,0,0,0,-.03,0)
        self.assertLess(boundary_f1(a,b,affine=affine,tolerance_m=.03)['F1'],boundary_f1(a,b,affine=affine,tolerance_m=.04)['F1'])
    def test_fixed_candidate_strata_and_coverage(self):
        a,b=mask(),mask(7,7);s=candidate_strata([[3,3]],[a,b],'F_frozen_hash',1)
        r=conditioned_recall([True,False],s);self.assertEqual(r['C1']['recall'],1);self.assertEqual(r['C0']['recall'],0)
    def test_count_bias_valid_union_not_tile_sum(self):
        from .spatial import valid_union_area
        tiles=[{'affine':(1,0,x,0,-1,50),'crs':'EPSG:32618','valid':np.ones((50,100),bool)} for x in (0,100)]
        cert=valid_union_area(tiles,10)
        r=count_statistics([2,3],[3,1],cert);self.assertEqual(r['count_MAE'],1.5);self.assertEqual(r['signed_bias'],-.5)
        with self.assertRaises(IntegrityError):count_statistics([1],[1],0)
    def test_proposal_coverage_does_not_equate_merge_with_detection(self):
        from .metrics import proposal_coverage
        a,b=mask(),mask(7,7);r=proposal_coverage([a,b],[a|b],True)
        self.assertEqual(r['all_proposal_coverage'],[1,1]);self.assertEqual(r['owner_linked_coverage'],[0,0])
        self.assertEqual(r['purity'],[.5]);self.assertEqual(r['ambiguous_owner_proposals'],1)
    def test_accuracy_coverage_curve_separates_abstention(self):
        from .metrics import accuracy_coverage_curve
        curve=accuracy_coverage_curve(np.array([[1,1,0],[0,0,1]],bool),[[0,1],[0,2]],[True,False],[.8,.9],[.5,.85],400)
        self.assertEqual(curve[0]['grouping']['pair_coverage'],1);self.assertEqual(curve[1]['grouping']['pair_coverage'],0)
    def test_pair_cache_owner_and_core_validation(self):
        from .metrics import validate_pair_cache
        a,b=mask(1,1,4,4),mask(7,7,4,4);valid=np.ones_like(a)
        r=validate_pair_cache([a,b],[10,20],valid,[[2,2],[3,3],[8,8]],[10,10,20],[[0,1],[0,2]],[True,False],'r1')
        self.assertEqual(r['eligible_GT'],2);self.assertEqual(r['same_pairs'],1)
        with self.assertRaises(IntegrityError):validate_pair_cache([a,b],[10,20],valid,[[2,2],[8,8]],[10,20],[[0,1]],[True],'r1')
    def test_fractional_indices_not_silently_rounded(self):
        with self.assertRaises(IntegrityError):candidate_strata([[1.7,2]],[mask()],'fixed',1)
        with self.assertRaises(IntegrityError):grouping(np.ones((1,3),bool),[[0,1.3]],[True])
    def test_union_area_deduplicates_non_square_overlap(self):
        from .spatial import valid_union_area
        tiles=[{'affine':(2,0,x,0,-1,2),'crs':'EPSG:32618','valid':np.ones((2,2),bool)} for x in (0,2)]
        self.assertEqual(valid_union_area(tiles,10)['valid_union_area_m2'],12) # tile sum16, overlap4
    def test_split_merge_overlap_graph_counts(self):
        a,b=mask(),mask(7,7)
        self.assertEqual(overlap_diagnostics([a,b],[a|b],.5)['merge_predictions'],1)
        self.assertEqual(overlap_diagnostics([a],[a,a],.5)['duplicate_extra_predictions'],1)
    def test_anchor_cores_report_eligibility(self):
        a=mask(1,1,8,8);counts=[core_mask(a,a,m)[0].sum() for m in ('r1','r4','adaptive')]
        self.assertGreater(counts[0],counts[1]);self.assertEqual(counts[1],0)

class CropOutputTests(unittest.TestCase):
    def test_epoch_determinism_validation_locked(self):
        def draw(ep,split):return rng_for(7,ep,split,'tile',2).integers(100000,size=10).tolist()
        self.assertEqual(draw(1,'train'),draw(1,'train'));self.assertNotEqual(draw(1,'train'),draw(2,'train'))
        self.assertEqual(draw(1,'validation'),draw(2,'validation'))
    def test_capacity_real_api_rejects_overflow(self):
        check_capacity(4,4)
        with self.assertRaises(IntegrityError):check_capacity(7,4)
    def test_complete_partial_background_invalid_support(self):
        a,b=mask(2,2,2,2),mask(6,6,5,5);v=np.ones_like(a);v[6:,6:]=0
        ts,r=record_crop([a,b],[17,23],v,(0,0,8,8),(.03,0,0,0,-.03,0),'synthetic')
        self.assertEqual(r['complete_annotation_ids'],[17]);self.assertEqual(r['partial_annotation_ids'],[23])
        self.assertEqual([t['annotation_id'] for t in ts],[17,23]);self.assertIn(23,r['outside_valid_annotation_ids'])
        self.assertIn('background',record_crop([],[],v,(0,0,8,8),(.03,0,0,0,-.03,0),'synthetic')[1]['categories'])
    def test_sampling_overflow_cannot_silently_drop(self):
        with self.assertRaises(IntegrityError):record_crop([mask(),mask(7,7)],[1,2],np.ones((12,12),bool),(0,0,12,12),(.03,0,0,0,-.03,0),'s',queries=1)
    def test_source_edge_crown_not_certified_complete_or_dropped(self):
        m=mask(0,0,3,3)
        targets,r=record_crop([m],[9],np.ones((12,12),bool),(0,0,12,12),(.03,0,0,0,-.03,0),'s')
        self.assertEqual(r['complete_count'],0);self.assertEqual(r['ambiguous_target_count'],1);self.assertEqual(len(targets),1)
    def test_category_planner_reports_infeasible_strata(self):
        from .crops import CATEGORIES,plan_crops
        quotas={k:0 for k in CATEGORIES};quotas['crowded']=1
        r=plan_crops([mask()],[1],np.ones((12,12),bool),(.03,0,0,0,-.03,0),(8,8),quotas,7,0,3)
        self.assertEqual(r['status'],'FAIL');self.assertIn('crowded',r['unmet_quotas'])
    def test_deterministic_sampler_and_impossible_valid_crop(self):
        v=np.ones((20,20),bool)
        self.assertEqual(choose_window(v,(8,8),5,1,'train','t',0,1),choose_window(v,(8,8),5,1,'train','t',0,1))
        with self.assertRaises(ValueError):choose_window(~v,(8,8),5,1,'train','t',0,1)
    def test_full_crown_fit_limitation_explicit(self):
        self.assertEqual(crown_centered_window(mask(1,1,9,9),(4,4),1)['status'],'DOES_NOT_FIT')
    def test_transform_image_mask_alignment_and_resize(self):
        m=mask();img=np.repeat(m[...,None].astype(np.uint8)*255,3,axis=2)
        im,ms=synchronized_transform(img,[m],k=1,flip_x=True)
        self.assertTrue(np.array_equal(im[...,0]>0,ms[0]))
        im,ms=synchronized_transform(img,[m],new_shape=(24,24));self.assertEqual(ms[0].sum(),m.sum()*4)
    def test_inactive_Q_not_materialized(self):
        logits=np.ones((3,4,4))
        p=materialize_q(logits,[.9,.1,.8],[.1,.9,.9],'t','n',.2,0)
        self.assertEqual(len(p),1);self.assertEqual(p[0].id,'t:q0')
    def test_output_lineage_validity_and_duplicate_selection(self):
        native={'n':{'stage':0,'input_sha256':'abc','config_sha256':'cfg'}};p=materialize_q(np.ones((2,4,4)),[.9,.8],[.1,.1],'t','n',.2,0)
        out=accept(p,np.ones((4,4),bool),(.03,0,0,0,-.03,0),.5,0,'exact_geometry_highest_score','keep_original',native)
        self.assertEqual(len(out),1);self.assertEqual(out[0].parents,('t:q0',));self.assertEqual(out[0].native_id,'n')
        with self.assertRaises(IntegrityError):validate(out,(3,3),2,native)
        with self.assertRaises(IntegrityError):validate(out,(4,4),2,{})
    def test_scene_stitch_lineage_order_and_global_coordinates(self):
        native={'n':{'stage':0,'input_sha256':'abc','config_sha256':'cfg'}}
        a=Instance('a',mask(1,1,2,2,(5,5)),.9,'n',('p',),2,'ta')
        b=Instance('b',mask(1,0,2,2,(5,5)),.8,'n',('p',),2,'tb')
        affine=(1,0,0,0,-1,5)
        tiles=[{'instances':[a],'shape':(5,5),'origin_yx':(0,0),'affine':affine,'crs':'EPSG:32618'}, {'instances':[b],'shape':(5,5),'origin_yx':(0,1),'affine':(1,0,1,0,-1,5),'crs':'EPSG:32618'}]
        s=stitch(tiles,'scene',(5,6),.5,'union',native,affine);t=stitch(tiles[::-1],'scene',(5,6),.5,'union',native,affine)
        self.assertEqual(len(s),1);self.assertEqual(s[0]['parents'],['a','b']);self.assertEqual(s[0]['id'],t[0]['id'])
        bad=[{**tiles[0],'affine':(2,0,0,0,-1,5)}]
        with self.assertRaises(IntegrityError):stitch(bad,'scene',(5,6),.5,'union',native,affine)
    def test_accumulation_flushes_partial_budget(self):
        from .runtime import accumulation_divisor,check_budget
        self.assertEqual([accumulation_divisor(i,5,2) for i in range(1,6)],[2,2,2,2,1])
        self.assertEqual(check_budget({'comparison':{'updates':6}},5,2,2),6)
    def test_GPU_budget_control_without_GPU_execution(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from .runtime import configure_cuda_budget
        props=SimpleNamespace(total_memory=10*1024**3,name='synthetic-device')
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.current_device',return_value=0),patch('torch.cuda.get_device_properties',return_value=props),patch('torch.cuda.set_per_process_memory_fraction') as setter:
            out=configure_cuda_budget({'comparison':{'GPU':'cuda:0','VRAM_GB':5}})
            setter.assert_called_once_with(.5,0);self.assertEqual(out['allocator_fraction'],.5)
    def test_buffered_calibration_excludes_close_neighbour(self):
        from .calibration import buffered_folds
        tiles=[{'id':str(i),'affine':(1,0,x,0,-1,5),'shape':[5,5],'crs':'EPSG:32618','role':'CALIBRATION'} for i,x in enumerate((0,5,100))]
        policy={'version':'fixture','support':'full_footprint','buffer_m':2,'approved':True,'buffer_justification':'synthetic fixture, not real spatial independence'}
        report=buffered_folds(tiles,{'0':0,'1':1,'2':1},policy)
        self.assertIn('1',report['folds'][0]['excluded_buffer_ids']);self.assertEqual(report['folds'][0]['train_ids'],['2'])
    def test_spatial_overlap_buffer_and_unknown_crossgrid(self):
        def tile(id,x,role):return {'id':id,'affine':(1,0,x,0,-1,5),'shape':(5,5),'crs':'EPSG:32618','role':role,'valid':np.ones((5,5),bool)}
        policy={'version':'fixture','support':'full_footprint','buffer_m':0,'approved':True,'buffer_justification':'fixture'}
        self.assertEqual(validate_split([tile('a',0,'TRAIN'),tile('b',1,'DEVELOPMENT')],policy,'confirmatory')['status'],'FAIL')
        self.assertEqual(validate_split([tile('a',0,'TRAIN'),tile('b',5,'DEVELOPMENT')],policy,'confirmatory')['status'],'PASS')
        policy={**policy,'support':'valid_pixels'}
        self.assertEqual(validate_split([tile('a',0,'TRAIN'),tile('b',.5,'DEVELOPMENT')],policy,'confirmatory')['status'],'UNKNOWN')

if __name__=='__main__':unittest.main()
