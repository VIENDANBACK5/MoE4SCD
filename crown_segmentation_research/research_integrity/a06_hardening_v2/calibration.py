"""Spatially buffered calibration folds with explicit excluded neighbours; no fitting here."""
from .core import canonical_hash,require
from .spatial import footprint,validate_split

def buffered_folds(tiles,assignment,policy):
    ids={t['id'] for t in tiles}
    require(set(assignment)==ids and len(set(assignment.values()))>=2,"Explicit complete multi-fold assignment required")
    require(policy.get('support')=='full_footprint',"Fold buffering currently supports conservative full footprints only")
    folds=[];blocked=[]
    for fold in sorted(set(assignment.values())):
        held=[t for t in tiles if assignment[t['id']]==fold];held_ids={t['id'] for t in held};train=[];excluded=[]
        for t in tiles:
            if t['id'] in held_ids:continue
            if any(t['crs']!=h['crs'] or footprint(t).intersection(footprint(h)).area>0 or footprint(t).distance(footprint(h))<policy['buffer_m'] for h in held):excluded.append(t['id'])
            else:train.append(t)
        checktiles=[{**t,'role':'CALIBRATION_TRAIN'} for t in train]+[{**t,'role':'CALIBRATION_HELDOUT'} for t in held]
        result=validate_split(checktiles,policy,'confirmatory')
        if not train or result['status']!='PASS':blocked.append(fold)
        folds.append({'fold':fold,'train_ids':[t['id'] for t in train],'heldout_ids':[t['id'] for t in held],'excluded_buffer_ids':excluded,'spatial_check':result})
    return {'version':'2.0.0','status':'FAIL' if blocked else 'PASS','blocked_folds':blocked,'folds':folds,
            'tiles':tiles,'assignment':assignment,'spatial_policy':policy,'assignment_hash':canonical_hash(assignment),
            'claims':'spatial-policy compliance only; does not certify ecological independence'}
