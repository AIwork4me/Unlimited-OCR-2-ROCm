"""Read-only GPU observations; same model arithmetic and original eager scope."""
import sys, json, hashlib, traceback
from pathlib import Path
from types import SimpleNamespace
R=Path.cwd(); O=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'experiments/cold_start'))
from worker import load, sha, write
te=load('reference_te',O/'te_worker.py')
import torch
sys.path.insert(0,str(R/'src'))
from unlimited_ocr_2_rocm import model
item=int(sys.argv[1]); declaration=json.loads((O/'predeclared.json').read_text('utf-8'))
condition=declaration['schedule'][item]['modes']
cache_source=Path(sys.prefix)/'Lib/site-packages/transformers/cache_utils.py'
assert sha(cache_source)==declaration['numerical_reference']['runtime']['cache_source_sha256']
out=O/f'process-{item}'/'data'; records=[]; request=[-1];samcalls=[0];steps=[0];handles=[]
active=[None];counts={};metadata=[];prefill=[]
decisions=[];cache_records=[];hook_counts={}
sys.path.insert(0,str(R/'experiments/numerical_reference'))
from diagnostic import clone_for_prefix
original=model.load_model
def save(name,t):
    if not isinstance(t,torch.Tensor):raise TypeError(name)
    cpu=t.detach().to('cpu',copy=True)
    p=out/'private-tensors'/f'{len(records):03d}.pt';p.parent.mkdir(parents=True,exist_ok=True);torch.save(cpu,p)
    records.append(dict(request=request[0],stage=name,shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device),sha256=hashlib.sha256(cpu.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),storage_offset=t.storage_offset(),storage_bytes=t.untyped_storage().nbytes(),file_sha256=sha(p),path=p.relative_to(out).as_posix()))
def loaded(*a,**kw):
    n=original(*a,**kw)
    module=__import__(n.model.__class__.__module__,fromlist=['_'])
    n.model._uocr_generate=clone_for_prefix(n.model._uocr_generate,declaration['numerical_reference']['common_prefix'],decisions,out/'teacher-forced-clone.py',guard_resolver=lambda:module._uocr_page_count_guard)
    guard=module._uocr_page_count_guard
    def observe_guard(*ga,**gkw):
        result=guard(*ga,**gkw)
        step=len(ga[1])
        if step in (7,10):save('logits.selection.'+str(step),result)
        return result
    module._uocr_page_count_guard=observe_guard
    block=n.model.model.vision_model.transformer.layers[0]
    def first(name,t):
        hook_counts[name]=hook_counts.get(name,0)+1
        if hook_counts[name]==1:save(name,t)
    handles.append(block.layer_norm1.register_forward_hook(lambda m,a,r:first('clip.norm1',r)))
    handles.append(block.self_attn.out_proj.register_forward_hook(lambda m,a,r:first('clip.attention_projected',r)))
    handles.append(block.layer_norm2.register_forward_pre_hook(lambda m,a:first('clip.residual',a[0])))
    handles.append(block.layer_norm2.register_forward_hook(lambda m,a,r:first('clip.norm2',r)))
    handles.append(block.mlp.fc2.register_forward_hook(lambda m,a,r:first('clip.mlp',r)))
    handles.append(block.register_forward_hook(lambda m,a,r:first('clip.block0.output',r)))
    def sam_pre(m,a):
        if samcalls[0]%2==0:request[0]+=1;steps[0]=0;counts.clear()
        samcalls[0]+=1;active[0]='sam';save('sam.input.'+str((samcalls[0]-1)%2),a[0])
    handles.append(n.model.model.sam_model.register_forward_pre_hook(sam_pre))
    def clip_pre(m,a):active[0]='clip'
    handles.append(n.model.model.vision_model.register_forward_pre_hook(clip_pre))
    for name in ['sam_model','vision_model','projector']:
        def post(m,a,r,_name=name):
            save(_name+'.output.'+str((samcalls[0]-1)%2),r)
            if _name in ('sam_model','vision_model'):active[0]=None
        handles.append(getattr(n.model.model,name).register_forward_hook(post))
    def decoder_post(m,a,r):
        if steps[0] in (0,7,10):
            step=steps[0];save('decoder.state.'+str(step),r.last_hidden_state[:,-1:])
            cache=r.past_key_values
            row=dict(step=step,type=type(cache).__name__,object_id=id(cache),layers=[])
            for layer in range(len(cache)):
                keys,values=cache[layer]
                entry=dict(layer=layer,sequence_length=cache.get_seq_length(layer),K_ptr=keys.data_ptr(),V_ptr=values.data_ptr())
                for name,t in [('K',keys),('V',values)]:
                    save('cache.'+name+'.'+str(step)+'.'+str(layer),t)
                    entry[name]=dict(shape=list(t.shape),dtype=str(t.dtype),stride=list(t.stride()),storage_offset=t.storage_offset())
                row['layers'].append(entry)
            cache_records.append(row)
    handles.append(n.model.model.register_forward_hook(decoder_post))
    def logits(m,a,r):
        step=steps[0];steps[0]+=1
        if step in (0,7,10):save(f'decoder.hidden.{step}',a[0]);save(f'logits.raw.{step}',r)
    handles.append(n.model.lm_head.register_forward_hook(logits))
    return n
class ObserveSDPA(torch.overrides.TorchFunctionMode):
    def __torch_function__(self,func,types,args=(),kwargs=None):
        kw=kwargs or {};sub=active[0]
        if func is torch.nn.functional.scaled_dot_product_attention and sub is not None:
            counts[sub]=counts.get(sub,0)+1
            if counts[sub]==1:
                for name,t in zip(['q','k','v'],args[:3]):save(sub+'.first_sdpa.'+name,t)
                mask=kw.get('attn_mask',args[3] if len(args)>3 else None)
                if mask is not None:save(sub+'.first_sdpa.mask',mask)
                metadata.append(dict(request=request[0],subsystem=sub,positional_count=len(args),non_tensor_kwargs={k:v for k,v in kw.items() if not isinstance(v,torch.Tensor)},mask_present=mask is not None))
                result=func(*args,**kw);save(sub+'.first_sdpa.output',result);return result
        return func(*args,**kw)
model.load_model=loaded
try:
    with ObserveSDPA():te.run(SimpleNamespace(output=out,declaration=O/'predeclared.json',item=item,option='eager',wheel='candidate',modes=condition,tensors=False,suite=False))
except BaseException:
    if out.exists():write(out/'capture-failure.json',{'traceback':traceback.format_exc()})
    raise
finally:
    model.load_model=original
    for h in handles:h.remove()
    if out.exists():write(out/'capture-index.json',dict(producer_sha256=sha(__file__),records=records,sdpa_metadata=metadata,prefill=prefill,sam_calls=samcalls[0],requests=request[0]+1,teacher_forced_decisions=decisions,cache=cache_records,runtime=dict(torch_version=torch.__version__,math_reduced_allowed=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),cache_source_sha256=sha(__import__('transformers.cache_utils',fromlist=['_']).__file__))))
