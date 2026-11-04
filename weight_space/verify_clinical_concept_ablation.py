"""Check direct-head ablation against a real last-step activation intervention."""
from pathlib import Path
import json
import numpy as np
import torch
from .data import load_data
from .run import Inference
from .checkpoints import sha256

@torch.inference_mode()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=load_data('mimic-iii-challenge_point');patient=int(data.indices('confirmation',32)[0]);batch=data.batch([patient])
 source=next(Path('../weights/ctimba_mimic-iii-challenge_point/version_0').glob('*.ckpt'));recipe=Path('weight_space/edits/clinical_v0_relative_tail.json');original_hash=sha256(source)
 runner=Inference(source,0,json.loads(recipe.read_text()),device='cuda',backend='triton')
 eff=(runner.model.out_proj[1].weight@runner.model.out_proj[0].mlp[3].weight).flatten()[47].cpu()
 phase=0;acts=[]
 def capture(module,inputs,output):
  nonlocal phase
  if phase%2:acts.append(output[...,47].cpu())
  phase+=1
 h=runner.model.out_proj[0].mlp[1].register_forward_hook(capture)
 p,draw=runner.predict(batch,[patient],100,[80,20.92],20261103,4,return_samples=True);h.remove()
 cout=runner.sigma_data*20.92/(20.92**2+runner.sigma_data**2)**.5
 predicted=draw.clone();delta=cout*torch.cat(acts)*eff
 predicted[0,:,:,:,0]=torch.where(batch['mask'][0,:,:,0],predicted[0,:,:,:,0],predicted[0,:,:,:,0]-delta)
 expected=predicted.median(1).values
 phase=0
 def intervene(module,inputs,output):
  nonlocal phase
  last=phase%2;phase+=1
  if last:
   output=output.clone();output[...,47]=0
  return output
 h=runner.model.out_proj[0].mlp[1].register_forward_hook(intervene)
 actual=runner.predict(batch,[patient],100,[80,20.92],20261103,4);h.remove()
 torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
 assert sha256(source)==original_hash
 out={'unit':47,'version':'V4','patients':1,'samples':100,'scope':'One implementation check on the first of the 32 confirmation patients; all 48x40 positions checked. Actual GELU ablation in final step versus algebraic draw correction and recomputed median. Not an accuracy improvement.','max_abs_normalized_prediction_difference':float((actual-expected).abs().max()),'rtol':2e-5,'atol':2e-6,'passed':True,'original_checkpoint_unchanged':True,'script_sha256':sha256(Path(__file__))}
 dest=Path('artifacts/audit/clinical_concept_ablation_verification_20261105.json');assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
