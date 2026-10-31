import json,os,sys,time
from collections import Counter
from pathlib import Path
R=Path.cwd();O=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'experiments/cold_start'))
from worker import sha,write
from windows_job import Job
from isolation_control import dispatch_valid
def read(p):return json.loads(Path(p).read_text('utf-8'))
a=read(O/'audit.json');assert all(all(v.values()) for k,v in a['checks'].items() if k!='wheels')
assert (R/'.b/te-e2e/numerical-reference-20261031-v1/retained-v2/retained-reference.json').exists()
for name in ['reference-green.log','diagnostic-tests.log','cache-tests.log','decorator-tests.log']:assert 'OK' in (O/name).read_text('utf-8')
old=R/'.b/te-e2e/sam-clip-isolation-20261031-v1'
d=read(old/'predeclared.json');d['schedule']=[dict(option='eager',repeat=0,modes=m) for m in ['MATH/MATH','AUTO/AUTO','AUTO/MATH','MATH/AUTO']]
baseline=read(R/'.b/te-e2e/qc26-mixed-v1/eager-MATH,MATH-0/data/worker/request-0/summary.json')
for p in [*O.glob('*.py'),* (R/'experiments/numerical_reference').glob('*.py')]:d['source_hashes'][p.relative_to(R).as_posix()]=sha(p)
d['worker_sha256']=sha(O/'te_worker.py');d['controller_sha256']=sha(__file__)
d['numerical_reference']=dict(runtime=read(O/'runtime-contract.json'),common_prefix=baseline['native']['token_ids'][:10],prefix_hash=sha(R/'.b/te-e2e/qc26-mixed-v1/eager-MATH,MATH-0/data/worker/request-0/summary.json'),requests=4,teacher_forcing='Record unchanged actual argmax, then feed common MM tokens0..9 into gen/cache/guard. Step10 unforced. Diagnostic, not native OCR parity qualification.',hypothesis='Matched CLIP QKV isolate MATH/flash output; common prefixes isolate logit7/10; supported DynamicCache indexing reveals KV hashes/positions.',stop='Any identity, dispatch, HIP, unchanged-parameter/buffer, completion/functional, cache position, original prefill/logit7 equivalence, or cleanup failure. No automatic expansion.',retained_reference_sha256=sha(R/'.b/te-e2e/numerical-reference-20261031-v1/retained-v2/retained-reference.json'))
write(O/'predeclared.json',d)
oldlabels=[R/'.b/te-e2e/mixed-rca-20261030-v1/fine-0/data',R/'.b/te-e2e/mixed-rca-20261030-v1/fine-1/data',old/'process-0/data',old/'process-2/data']
for i,item in enumerate(d['schedule']):
    folder=O/f'process-{i}';folder.mkdir();tmp=folder/'tmp';tmp.mkdir()
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONHASHSEED='0',TEMP=str(tmp),TMP=str(tmp),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRITON_CACHE_DIR=str(R/'.b/g1/triton'),PYTORCH_KERNEL_CACHE_PATH=str(R/'.b/te-e2e/kernels'),TORCH_EXTENSIONS_DIR=str(R/'.b/te-e2e/extensions'))
    job=Job();command=[str(R/'.b/o/Scripts/python.exe'),'-u',str(O/'capture.py'),str(i)];receipt=dict(command=command,condition=item['modes'],start=time.time())
    try:
        with (folder/'stdout.log').open('wb') as so,(folder/'stderr.log').open('wb') as se:
            p=job.spawn(command,cwd=R,env=env,stdout=so,stderr=se);receipt.update(pid=p.pid,exit_code=p.wait(timeout=1200))
    finally:
        job.terminate();receipt['accounting']=job.accounting();job.close();receipt['end']=time.time();write(folder/'finished.json',receipt)
    assert receipt['exit_code']==0 and receipt['accounting']['active_processes']==0
    data=folder/'data';summary=read(data/'worker/request-0/summary.json');ob=read(data/'worker/observation-0.json');idx=read(data/'capture-index.json');observer=read(data/'observers.json')
    assert read(data/'worker/complete.json')==dict(status='COMPLETED',requests=1)
    assert idx['requests']==1 and idx['sam_calls']==2 and len(idx['records'])==106
    expected=[f'{n}.{k}' for k in range(2) for n in ['sam.input','sam_model.output','vision_model.output','projector.output']]
    expected+=['sam.first_sdpa.'+n for n in ['q','k','v','mask','output']]+['clip.first_sdpa.'+n for n in ['q','k','v','output']]
    expected+=['clip.'+n for n in ['norm1','attention_projected','residual','norm2','mlp','block0.output']]
    expected+=[f'{n}.{step}' for step in [0,7,10] for n in ['decoder.state','decoder.hidden','logits.raw']]+['logits.selection.7','logits.selection.10']
    expected+=[f'cache.{n}.{step}.{layer}' for step in [0,7,10] for layer in range(12) for n in ['K','V']]
    assert Counter(r['stage'] for r in idx['records'])==Counter(expected)
    assert len(idx['sdpa_metadata'])==2 and all(m['positional_count']==3 for m in idx['sdpa_metadata'])
    assert {m['subsystem']:m['mask_present'] for m in idx['sdpa_metadata']}=={'sam':True,'clip':False}
    guard=[json.loads(line) for line in (data/'worker/request-0/diagnostics/guard.jsonl').read_text('utf-8').splitlines()]
    assert all(guard[step]['step']==step and guard[step]['branch']=='unmodified' for step in [7,10])
    decisions=idx['teacher_forced_decisions'];assert len(decisions)>=11 and [x['step'] for x in decisions[:11]]==list(range(11))
    assert [x['fed_token'] for x in decisions[:10]]==d['numerical_reference']['common_prefix'] and len(d['numerical_reference']['common_prefix'])==10
    assert decisions[10]['actual_selected']==decisions[10]['fed_token']==summary['native']['token_ids'][10]
    assert dispatch_valid(ob['dispatch'],item['modes']) and summary['status']==summary['correctness']['status']=='PASS'
    assert summary['validated_metadata']['finish_reason']=='stop' and summary['runtime_state_restored']
    assert all(summary[k]['status']=='PASS' for k in ['hip_before','hip_after'])
    assert all(ob['before'][k]==ob['after'][k] for k in ['parameters','buffers'])
    assert observer['te_before'] is observer['te_after'] is observer['graph_optimize_after'] is True
    assert summary['effective_options']==baseline['effective_options'] and summary['native']['prompt_token_ids']==baseline['native']['prompt_token_ids']
    assert summary['native']['token_ids'][:10]==d['numerical_reference']['common_prefix']
    assert [r['step'] for r in idx['cache']]==[0,7,10]
    assert all(len(r['layers'])==12 and all(l['sequence_length']==len(summary['native']['prompt_token_ids'])+r['step'] for l in r['layers']) for r in idx['cache'])
    oldidx=read(oldlabels[i]/'capture-index.json');oldmap={r['stage']:r for r in oldidx['records'] if r['request']==0};newmap={r['stage']:r for r in idx['records']}
    assert all(newmap[n]['sha256']==oldmap[n]['sha256'] for n in ['sam.input.0','sam_model.output.0','vision_model.output.0','decoder.hidden.0','logits.raw.7'])
    write(folder/'admission.json',dict(physical_functional=True,teacher_prefix=True,cache_positions=True,old_prefill_and_step7_exact=True))
    print('finished',i,item['modes'],flush=True)
