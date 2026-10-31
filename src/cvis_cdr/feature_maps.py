"""Early ResNet18 block (layer1[0]) activation visualization; not original Fig 7 code."""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from .experiment import model_init
from .records import read_array
from .processing import image_rgb

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--sample',required=True)
 p.add_argument('--kind',default='rd_db');p.add_argument('--out',default='figure7_features_reconstructed.png');a=p.parse_args()
 ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 model=model_init(ckpt['nclasses']);model.load_state_dict(ckpt['model']);model.eval()
 rd=read_array(a.sample,a.kind)
 rgb=image_rgb(rd,*ckpt['bounds'],size=ckpt['input_size'])
 outputs=[];handle=model.layer1[0].register_forward_hook(lambda _m,_inp,out:outputs.append(out.detach().cpu()))
 with torch.inference_mode():model(torch.from_numpy(rgb.copy()).unsqueeze(0))
 handle.remove()
 heat=outputs[0][0].abs().mean(dim=0).numpy()
 fig,ax=plt.subplots(1,2,figsize=(8,3.5));ax[0].imshow(rgb.transpose(1,2,0));ax[0].set_title('RD RGB input');ax[1].imshow(heat,cmap='viridis');ax[1].set_title('Layer1[0] mean |feature|')
 for axis in ax:axis.axis('off')
 fig.tight_layout();Path(a.out).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=180);plt.close(fig)
 print('Saved:',a.out)
if __name__=='__main__':main()
