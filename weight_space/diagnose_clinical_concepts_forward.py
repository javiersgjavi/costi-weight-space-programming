"""Bounded paired V1/V4 concept diagnostics. Frozen weights, no gradients/fitting.

Store activation/clinical arrays privately; export only aggregate interventions.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .run import Inference
from .data import load_data
from .checkpoints import sha256
from .analyze_clinical_neuron_concepts import NAMES,UNITS

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--chunk',type=int,default=4);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=load_data('mimic-iii-challenge_point');ids=data.indices('confirmation',32)
 source=next(Path('../weights/ctimba_mimic-iii-challenge_point/version_0').glob('*.ckpt'))
 recipes={'V1':Path('weight_space/edits/clinical_v0_affine_tail_readout.json'),'V4':Path('weight_space/edits/clinical_v0_relative_tail.json')}
 # Fix input interventions before executing models. Two eligible patients per
 # variable from the same 32, with >=2 conditioning observations and scored entries.
 interventions={}
 train_rows=(data.starts[data.train,None]+np.arange(48)).reshape(-1)
 for node in [0,1,21,19]:
  values=data.values[train_rows,node][data.observed[train_rows,node]]
  physical_shift=.5*float(np.std(values,dtype=np.float64));delta=physical_shift/float(data.scale[0,0,node,0])
  eligible=[]
  for patient in ids:
   b=data.batch([patient])
   if b['mask'][0,:,node,0].sum()>=2 and b['eval_mask'][0,:,node,0].any():eligible.append(int(patient))
  chosen=[eligible[i] for i in np.linspace(0,len(eligible)-1,min(2,len(eligible)),dtype=int)] if eligible else []
  for patient in chosen:interventions.setdefault(patient,[]).append((node,delta,physical_shift))
 plan={'dataset':data.name,'split':'confirmation','patients':32,'samples':100,'seed':20261103,'sigmas':[80,20.92],'chunk':a.chunk,'source_sha256':sha256(source),'recipe_sha256':{k:sha256(v) for k,v in recipes.items()},'input_intervention':'Raise conditioning observations of one variable by 0.5 SD of observed training values, preserving mask; recompute its interpolation by the same translation. Synthetic sensitivity probe, not a clinical intervention.','intervention_variables':[NAMES[x] for x in [0,1,21,19]],'intervention_patient_variable_cases':sum(map(len,interventions.values())),'comparison':'Paired same patients/noise; summarize all-draw preactivations for input effects to avoid median selection confounding. Scored median-selected activations saved for descriptive associations.','ablation':'Exact last-step-only removal of individual/group head contributions across ALL 100 draws followed by a new median. Not full two-step ablation and not a candidate.','gradients_training_or_weight_updates':False,'source_script_sha256':sha256(Path(__file__))}
 (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 for version,recipe_path in recipes.items():
  runner=Inference(source,0,json.loads(recipe_path.read_text()),device='cuda',backend='triton')
  effective=(runner.model.out_proj[1].weight@runner.model.out_proj[0].mlp[3].weight).flatten().cpu().numpy()
  records={k:[] for k in ['prediction','residual','node','window','median_selected_draw','hidden','preactivation','hidden_step1','preactivation_step1']}
  ablations=[];effects=[]
  captures=[[],[]];all_z=[[],[]];count=0;scoring=None
  def hook(module,inputs,output):
   nonlocal count
   step=count%2;z=inputs[0]
   captures[step].append((z[:,scoring].cpu(),output[:,scoring].cpu()))
   all_z[step].append(z[:,:,:,[38,198,47,255]].sum(0).cpu())
   count+=1
  handle=runner.model.out_proj[0].mlp[1].register_forward_hook(hook)
  def run(b,patient):
   nonlocal count,scoring
   captures[0].clear();captures[1].clear();all_z[0].clear();all_z[1].clear();count=0
   scoring=b['eval_mask'][0,:,:,0].cuda()
   pred,draws=runner.predict(b,[patient],100,[80,20.92],20261103,a.chunk,return_samples=True)
   zz=[torch.cat([x[0] for x in c]) for c in captures];hh=[torch.cat([x[1] for x in c]) for c in captures]
   means=[torch.stack(c).sum(0).numpy()/100 for c in all_z]
   return pred,draws,zz,hh,means
  start=time.monotonic()
  try:
   for number,patient in enumerate(ids):
    b=data.batch([patient]);mask=b['eval_mask'];s=mask[0,:,:,0]
    pred,draws,zz,hh,means=run(b,int(patient));scored=draws[0,:,:,:,0][:,s];sel=scored.median(0).indices;entry=torch.arange(len(sel))
    nodes=torch.arange(40).expand(48,-1)[s];truth=((b['target']-data.bias)/data.scale)[mask]
    records['prediction'].append(pred[mask].numpy());records['residual'].append((pred[mask]-truth).numpy());records['node'].append(nodes.numpy());records['window'].append(np.full(len(sel),patient));records['median_selected_draw'].append(sel.numpy())
    for step,suffix in [(0,'_step1'),(1,'')]:
     records['hidden'+suffix].append(hh[step][sel,entry].numpy());records['preactivation'+suffix].append(zz[step][sel,entry].numpy())
    cout=runner.sigma_data*20.92/(20.92**2+runner.sigma_data**2)**.5
    for label,units in [('38',[38]),('198',[198]),('47',[47]),('255',[255]),('midrange198',[246,226,179]),('contrast38_198',[181,150])]:
     contribution=(hh[1][:,:,units]*torch.from_numpy(effective[units])).sum(-1)
     changed=(scored-cout*contribution).median(0).values
     delta=(changed-pred[mask]).numpy()
     for node in np.unique(nodes.numpy()):
      m=nodes.numpy()==node;ablations.append({'group':label,'variable':NAMES[node],'entries':int(m.sum()),'sum_abs_delta':float(np.abs(delta[m]).sum(dtype=np.float64)),'sum_delta':float(delta[m].sum(dtype=np.float64))})
    for node,delta,physical in interventions.get(int(patient),[]):
     changed={k:v.clone() for k,v in b.items()};obs=changed['mask'][0,:,node,0]
     changed['x'][0,:,node,0]+=obs*delta;changed['x_interpolated'][0,:,node,0]+=delta
     pred2,_,_,_,means2=run(changed,int(patient));targetmask=s[:,node].numpy()
     for step in [0,1]:
      diff=means2[step]-means[step]
      effects.append({'variable':NAMES[node],'step':step+1,'entries':int(targetmask.sum()),'input_shift_normalized':delta,'input_shift_physical':physical,'mean_delta_z':diff[:,node,:][targetmask].mean(0).astype(float).tolist(),'mean_abs_delta_z_other_variables':np.abs(np.delete(diff,node,axis=1)).mean((0,1)).astype(float).tolist(),'mean_prediction_delta_normalized':float((pred2-pred)[0,:,node,0][s[:,node]].mean())})
    print(json.dumps({'version':version,'done':number+1,'total':len(ids),'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
  finally:handle.remove()
  arrays={k:np.concatenate(v) for k,v in records.items()};arrays['effective']=effective
  np.savez_compressed(a.output/f'{version}_private_activations.npz',**arrays)
  agg=[]
  for group in sorted({x['group'] for x in ablations}):
   for variable in NAMES:
    rows=[x for x in ablations if x['group']==group and x['variable']==variable]
    if not rows:continue
    n=sum(r['entries'] for r in rows)
    agg.append({'group':group,'variable':variable,'entries':n,'patients':len(rows),'mean_abs_prediction_delta_normalized':sum(r['sum_abs_delta'] for r in rows)/n,'mean_prediction_delta_normalized':sum(r['sum_delta'] for r in rows)/n})
  result={'version':version,'protocol':plan,'input_effects':effects,'ablation_by_variable':agg,'private_activation_sha256':sha256(a.output/f'{version}_private_activations.npz'),'weights_unchanged':True}
  (a.output/f'{version}_aggregate.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
  del runner;torch.cuda.empty_cache()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
