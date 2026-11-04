"""Forward-free descriptive concept analysis. No probes are fitted; no gradients.

Private activation and clinical data stay local. Only aggregated summaries are exported.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr
from .data import load_data
from .checkpoints import sha256

NAMES='HR O2Sat Temp SBP MAP DBP Resp EtCO2 BaseExcess HCO3 FiO2 pH PaCO2 SaO2 AST BUN Alkalinephos Calcium Chloride Creatinine Bilirubin_direct Glucose Lactate Magnesium Phosphate Potassium Bilirubin_total TroponinI Hct Hgb PTT WBC Fibrinogen Platelets Age Gender Unit1 Unit2 HospAdmTime ICULOS'.split()
UNITS=[38,198,47,255,246,226,179,181,150]

def rho(a,b):
 if len(a)<5 or np.ptp(a)==0 or np.ptp(b)==0:return None
 return float(spearmanr(a,b).statistic)

def within_variable_rank(a,nodes):
 out=np.zeros(len(a))
 for node in np.unique(nodes):
  m=nodes==node;n=int(m.sum());out[m]=(rankdata(a[m])-(n+1)/2)/n
 return out

def features(dataset,windows,nodes):
 out={k:[] for k in ['time','interpolation','target','observation_density','nearest_observation','slope','absolute_slope','temporal_variability']}
 rows_nodes=[];rows_windows=[];physical_targets=[]
 for patient in np.unique(windows):
  b=dataset.batch([patient]);m=b['eval_mask'][0,:,:,0].numpy();obs=b['mask'][0,:,:,0].numpy();itp=b['x_interpolated'][0,:,:,0].numpy();target=((b['target']-dataset.bias)/dataset.scale)[0,:,:,0].numpy()
  slope=np.gradient(itp,axis=0);distance=np.full((48,40),48.)
  for node in range(40):
   ts=np.flatnonzero(obs[:,node])
   if len(ts):distance[:,node]=np.min(np.abs(np.arange(48)[:,None]-ts[None,:]),axis=1)
  mats={'time':np.broadcast_to(np.arange(48)[:,None],(48,40)), 'interpolation':itp,'target':target,'observation_density':np.broadcast_to(obs.mean(0),(48,40)), 'nearest_observation':distance,'slope':slope,'absolute_slope':np.abs(slope),'temporal_variability':np.broadcast_to(itp.std(0),(48,40))}
  for key,val in mats.items():out[key].append(val[m])
  physical_targets.append(b['target'][0,:,:,0].numpy()[m])
  rows_nodes.append(np.broadcast_to(np.arange(40),(48,40))[m]);rows_windows.append(np.full(m.sum(),patient))
 assert np.array_equal(np.concatenate(rows_nodes),nodes)
 assert np.array_equal(np.concatenate(rows_windows),windows)
 result={k:np.concatenate(v) for k,v in out.items()}
 if not hasattr(dataset,'concept_observed_train_cdf'):
  train_rows=(dataset.starts[dataset.train,None]+np.arange(48)).reshape(-1)
  dataset.concept_observed_train_cdf={}
  for node in range(40):
   vals=dataset.values[train_rows,node][dataset.observed[train_rows,node]]
   dataset.concept_observed_train_cdf[node]=np.sort(vals)
 for key in ['target','interpolation']:
  rank=np.zeros(len(nodes))
  for node in np.unique(nodes):
   m=nodes==node;vals=dataset.concept_observed_train_cdf[node]
   # Preserve exact observed-value ties for target percentiles; do not
   # round-trip a float32 target through the scaler.
   physical=np.concatenate(physical_targets)[m] if key=='target' else result[key][m]*float(dataset.scale[0,0,node,0])+float(dataset.bias[0,0,node,0])
   rank[m]=(np.searchsorted(vals,physical,'left')+np.searchsorted(vals,physical,'right'))/(2*len(vals))
  result[key+'_observed_training_percentile']=rank
 return result

def summarize(arr,dataset):
 nodes,windows=arr['node'],arr['window'];f=features(dataset,windows,nodes)
 assert np.allclose(f['target'],arr['prediction']-arr['residual'],rtol=2e-5,atol=3e-6)
 f['prediction']=arr['prediction'];f['prediction_minus_interpolation']=arr['prediction']-f['interpolation']
 ranks={k:within_variable_rank(v,nodes) for k,v in f.items()}
 report={'scored_entries':len(nodes),'patients':len(np.unique(windows)),'units':{},'feature_definitions':{'interpolation':'Observed-only temporal interpolation, normalized by original training scaler','target':'Held-out known target, descriptive association only','observation_density':'Observed conditioning entries / 48 in same patient and variable','nearest_observation':'Distance in hours to nearest conditioning observation; 48 if absent','slope':'Signed numpy.gradient of observed-only normalized interpolation','temporal_variability':'SD of interpolated signal over 48 hours','within_variable_spearman':'Pearson correlation of centered percentile ranks within each variable, pooled over entries; descriptive, not a fitted probe'}}
 for step,suf in [(2,''),(1,'_step1')]:
  if 'hidden'+suf not in arr:continue
  for j in UNITS:
   h=arr['hidden'+suf][:,j];c=h*arr['effective'][j];z=arr.get('preactivation'+suf)
   label=f'{j}_step{step}';active=(z[:,j]>0) if z is not None else (h>0)
   ranked=within_variable_rank(c,nodes)
   correlations={k:{'pooled_spearman':rho(c,v),'within_variable_spearman':float(np.corrcoef(ranked,ranks[k])[0,1]) if np.std(ranked)>0 and np.std(ranks[k])>0 else None} for k,v in f.items()}
   var=[]
   for n in np.unique(nodes):
    m=nodes==n
    var.append({'variable':NAMES[n],'entries':int(m.sum()),'patients':int(len(np.unique(windows[m]))),'positive_preactivation_fraction':float(active[m].mean()),'mean_abs_contribution':float(np.abs(c[m]).mean()),'contribution_vs_interpolation_spearman':rho(c[m],f['interpolation'][m]),'contribution_vs_target_spearman':rho(c[m],f['target'][m])})
   row={'unit':j,'step':step,'effective_coefficient':float(arr['effective'][j]),'positive_preactivations':int(active.sum()),'positive_patient_count':int(len(np.unique(windows[active]))),'positive_physiological_fraction':float((nodes[active]<34).mean()) if active.any() else None,'mean_abs_contribution':float(np.abs(c).mean()),'correlations':correlations,'variables':var}
   if z is not None:
    row['z_quantiles']=np.quantile(z[:,j],[0,.01,.1,.5,.9,.99,1]).tolist()
    zr=within_variable_rank(z[:,j],nodes)
    row['preactivation_correlations']={k:{'pooled_spearman':rho(z[:,j],v),'within_variable_spearman':float(np.corrcoef(zr,ranks[k])[0,1]) if np.std(zr)>0 and np.std(ranks[k])>0 else None} for k,v in f.items()}
   if active.any():
    row['positive_region']={'median_normalized_target':float(np.median(f['target'][active])),'fraction_target_above_2_normalized':float((f['target'][active]>2).mean()),'median_target_observed_training_percentile':float(np.median(f['target_observed_training_percentile'][active])),'fraction_target_above_95th_observed_training_percentile':float((f['target_observed_training_percentile'][active]>.95).mean())}
   report['units'][label]=row
 if 'preactivation' in arr:
  d=arr['preactivation'][:,38]-arr['preactivation'][:,198];m=d>0
  report['contrast']={'positive_entries':int(m.sum()),'patients':int(len(np.unique(windows[m]))),'variable_counts':{NAMES[n]:int(((nodes==n)&m).sum()) for n in np.unique(nodes[m])},'median_normalized_target':float(np.median(f['target'][m])),'median_normalized_interpolation':float(np.median(f['interpolation'][m])),'median_nearest_observation_hours':float(np.median(f['nearest_observation'][m])),'median_observation_density':float(np.median(f['observation_density'][m])),'fraction_target_gt_interpolation':float((f['target'][m]>f['interpolation'][m]).mean()),'mean_normalized_residual':float(arr['residual'][m].mean())} if m.any() else {'positive_entries':0}
 if 'contrast' in report and report['contrast']['positive_entries']:
  report['contrast']['median_target_observed_training_percentile']=float(np.median(f['target_observed_training_percentile'][m]))
  report['contrast']['fraction_target_above_95th_observed_training_percentile']=float((f['target_observed_training_percentile'][m]>.95).mean())
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--diagnostic',type=Path,action='append',required=True);args=p.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 data=load_data('mimic-iii-challenge_point');result={'scope':'Exploratory, descriptive concept associations; no causal or clinical specificity claim. Separate cohort sizes must not be interpreted as paired improvement.','gradients_or_training_used':False,'diagnostics':{}}
 for path in args.diagnostic:
  with np.load(path,allow_pickle=False) as src:arr={k:src[k] for k in src.files}
  d=summarize(arr,data);d['source_sha256']=sha256(path);result['diagnostics'][path.stem]=d
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for label,d in result['diagnostics'].items():
  print(label,d['scored_entries'],d['patients'])
  for u in [38,198,47,255]:
   row=d['units'][f'{u}_step2'];print(u,'active',row['positive_preactivations'],'coeff',round(row['effective_coefficient'],4),'correlations',{k:round(v['within_variable_spearman'],3) if v['within_variable_spearman'] is not None else None for k,v in row['correlations'].items()})
  if 'contrast'in d:print('contrast',d['contrast'])

if __name__=='__main__':main()
