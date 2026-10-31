"""Reconstructed sweep, runs predefined methods/seeds. Not archived prior sweep."""
import argparse,subprocess,sys

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--task',choices=['binary','counting'],default='counting');p.add_argument('--seeds',type=int,nargs='+',default=[42]);p.add_argument('--methods',nargs='+',default=['naive','random_dr','cdr']);p.add_argument('--epochs',type=int,default=30);p.add_argument('--input-size',type=int,default=224);a=p.parse_args()
 for m in a.methods:
  for seed in a.seeds:
   cmd=[sys.executable,'-m','cvis_cdr.experiment','train','--manifest',a.manifest,'--task',a.task,'--method',m,'--seed',str(seed),'--epochs',str(a.epochs),'--input-size',str(a.input_size)]
   print('RUN', ' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
if __name__=='__main__':main()
