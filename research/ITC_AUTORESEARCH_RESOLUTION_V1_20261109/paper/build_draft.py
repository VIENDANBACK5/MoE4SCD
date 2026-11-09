"""Build manuscript tables directly from executed records; no new scoring."""
from pathlib import Path
import json,csv,hashlib,re

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'; EXP=ROOT/'experiments/frozen_bam_transfer'
def rows(name):
    with (EXP/'results'/name).open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    m=rows('METRICS.csv'); paired=rows('RR_PAIRED_RESULTS.csv'); same=rows('SAME_COUNT_RESULTS.csv')
    dec=json.loads((EXP/'SCIENTIFIC_DECISION.json').read_text())
    assert len(m)==9 and all(int(x['GT'])==1106 and not x['original_source_small_Recall50'] for x in m)
    assert sum(int(x['new']) for x in paired)==58 and sum(int(x['lost']) for x in paired)==43
    assert sum(int(x['delta_TP50']) for x in paired)==15
    for x in paired:assert int(x['new'])-int(x['lost'])==int(x['delta_TP50'])
    assert abs(sum(float(x['delta_Recall50_pp']) for x in paired)/3-dec['mean_RR_delta']['delta_Recall50_pp'])<1e-10
    by={(x['seed'],x['arm']):x for x in same}
    sc=sum(float(by[(str(s),'RR')]['Recall50'])-float(by[(str(s),'CONTROL')]['Recall50']) for s in [42,43,44])/3
    assert abs(sc-dec['same_count_delta_recall_pp'])<1e-10
    tab=['| System | AP | AP75 | Recall50 % | Recall75 % | TP50 | FP50 | Raw50 |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in m:
        tab.append('| '+x['arm']+' | '+' | '.join(f"{float(x[k]):.3f}" for k in ['AP','AP75','Recall50','Recall75'])+' | '+' | '.join(x[k] for k in ['TP50','FP50','raw50'])+' |')
    pt=['| Seed | ΔAP | ΔRecall50 pp | New | Lost | Net TP50 | ΔFP50 |','|---|---:|---:|---:|---:|---:|---:|']
    for x in paired:
        pt.append(f"| {x['seed']} | {float(x['delta_AP']):+.3f} | {float(x['delta_Recall50_pp']):+.3f} | {x['new']} | {x['lost']} | {int(x['delta_TP50']):+d} | {int(x['delta_FP50']):+d} |")
    text=(PAPER/'PAPER_TEMPLATE.md').read_text().replace('{{FINAL_RESULTS}}','\n'.join(tab)).replace('{{PAIRED_RESULTS}}','\n'.join(pt))
    assert '{{' not in text
    (PAPER/'PAPER_DRAFT.md').write_text(text)
    # Local reading copy; equations remain visible TeX rather than fetching scripts.
    import markdown
    body=markdown.markdown(text,extensions=['tables','fenced_code'])
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Frozen cross-carrier ITC evidence draft</title><style>body{font:17px/1.6 Georgia,serif;max-width:1080px;margin:40px auto;padding:0 22px;color:#18212b}h1,h2,h3{font-family:system-ui,sans-serif;line-height:1.25}table{border-collapse:collapse;font:14px system-ui;width:100%;margin:20px 0}td,th{padding:8px;border:1px solid #ccd2d7;text-align:right}th:first-child,td:first-child{text-align:left}a{color:#175b91}pre,code{white-space:pre-wrap}img{max-width:100%}</style>'+body+'</html>'
    (PAPER/'PAPER_DRAFT.html').write_text(page)
    broken=[];checked=[]
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if '://' not in target:
            p=PAPER/target;checked.append(target)
            if not p.exists():broken.append(target)
    assert not broken,broken
    sources=['METRICS.csv','RR_PAIRED_RESULTS.csv','SAME_COUNT_RESULTS.csv','RECOVERY_AND_LOSS.csv','CROSS_CARRIER_PATTERNS.csv']
    status=dict(status='EXECUTED_RESULTS_DRAFT_NOT_ORIGINAL_METHOD_OR_SUBMISSION_READY',
                manuscript_sha256=sha(PAPER/'PAPER_DRAFT.md'),html_sha256=sha(PAPER/'PAPER_DRAFT.html'),tables_generated_from_actual_CSV=True,
                arithmetic_and_same_count_checks=True,local_links_checked=len(checked),broken_local_links=broken,
                source_hashes={name:sha(EXP/'results'/name) for name in sources},
                actual_independent_reviewer_responses=0,new_model_forwards=0,goal_complete=False)
    (PAPER/'DRAFT_VERIFICATION.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status))

if __name__=='__main__':main()
