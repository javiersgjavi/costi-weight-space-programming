"""Build a public aggregate report without patient identifiers or activation rows."""
import json
from pathlib import Path
import numpy as np
from .checkpoints import sha256

ROOT=Path('artifacts/audit')
def main():
 calpath=ROOT/'clinical_neuron_concepts_exact_percentiles_20261105.json';confpath=calpath
 cal=json.loads(calpath.read_text())['diagnostics'];conf=json.loads(confpath.read_text())['diagnostics']
 units=[38,198,47,255];features=['interpolation','slope','temporal_variability','observation_density','nearest_observation','time']
 out={'scope':'Exploratory functional interpretation, not clinical concept identification. No training, fitted probe, gradients, checkpoint updates or new full benchmark evaluation.','units':units,'features':features,'calibration':{},'confirmation':{},'input_interventions':{},'last_step_ablations':{},'sources':{str(calpath):sha256(calpath),str(confpath):sha256(confpath)},'evidence_limits':['Calibration cohorts V1=128 and V4=512 are different; paired new confirmation uses the same 32 patients and noise.','Cohorts were used previously in the broader benchmark; no untouched external holdout is claimed.','Within-variable ranks remove between-variable level differences but do not remove all confounding; entries within a patient are dependent.','Input interventions have only two eligible patients per variable, selected before inference; descriptive sensitivities only.','Observed training percentiles are empirical reference ranks, not clinical normality thresholds.','Ablations remove last-step contributions only and recompute all-draw median; they do not establish the effect of removing a unit across the entire model.','A lack of a marginal correlation does not exclude a nonlinear or context-dependent concept.']}
 for version,label in [('V1','clinical_targeted_head_cal128_median100_v0'),('V4','clinical_relative_tail_head_both_steps_cal512')]:
  d=cal[label];out['calibration'][version]={'patients':d['patients'],'entries':d['scored_entries'],'positive_regions':{str(j):d['units'][f'{j}_step2'].get('positive_region') for j in units},'positive_counts':{str(j):d['units'][f'{j}_step2']['positive_preactivations'] for j in units},'contrast':d.get('contrast')}
  out['calibration'][version]['effective_coefficients']={str(j):d['units'][f'{j}_step2']['effective_coefficient'] for j in units}
  if version=='V4':out['calibration'][version]['unit_variable_profiles']={str(j):d['units'][f'{j}_step2']['variables'] for j in units}
  d=conf[version+'_private_activations'];out['confirmation'][version]={'patients':d['patients'],'entries':d['scored_entries'],'matrix':[[d['units'][f'{j}_step2']['preactivation_correlations'][f]['within_variable_spearman'] for f in features] for j in units],'positive_regions':{str(j):d['units'][f'{j}_step2'].get('positive_region') for j in units},'contrast':d.get('contrast')}
  path=ROOT/f'clinical_concepts_confirmation32_20261105/{version}_aggregate.json';raw=json.loads(path.read_text());out['sources'][str(path)]=sha256(path)
  out['input_interventions'][version]=[]
  for variable in ['HR','O2Sat','Glucose','Creatinine']:
   for step in [1,2]:
    rows=[r for r in raw['input_effects'] if r['variable']==variable and r['step']==step]
    if rows:
     z=np.array([r['mean_delta_z'] for r in rows]);out['input_interventions'][version].append({'variable':variable,'step':step,'patient_cases':len(rows),'mean_delta_preactivation':z.mean(0).tolist(),'fraction_cases_positive_delta':(z>0).mean(0).tolist(),'mean_prediction_delta_normalized':float(np.mean([r['mean_prediction_delta_normalized'] for r in rows]))})
  out['last_step_ablations'][version]=[]
  for group in sorted({r['group'] for r in raw['ablation_by_variable']}):
   rows=[r for r in raw['ablation_by_variable'] if r['group']==group];n=sum(r['entries'] for r in rows)
   out['last_step_ablations'][version].append({'group':group,'entries':n,'mean_absolute_prediction_delta_normalized':sum(r['entries']*r['mean_abs_prediction_delta_normalized'] for r in rows)/n,'mean_signed_prediction_delta_normalized':sum(r['entries']*r['mean_prediction_delta_normalized'] for r in rows)/n})
 out['ablation_verification']=json.loads((ROOT/'clinical_concept_ablation_verification_20261105.json').read_text())
 dest=ROOT/'clinical_neuron_concepts_summary_20261105.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(dest)
if __name__=='__main__':main()
