import argparse,os,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description='Standalone house detection + unique target service')
 p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=18081)
 p.add_argument('--gpu',default='0',help='Physical GPU index, GPU UUID, or cpu')
 p.add_argument('--weights',type=Path,default=ROOT/'trained_model/house_v3_recommended.pt')
 a=p.parse_args()
 if not 1<=a.port<=65535:p.error('port must be 1..65535')
 if a.gpu!='cpu' and not re.fullmatch(r'(\d+|GPU-[0-9a-fA-F-]+)',a.gpu):p.error('gpu must be one GPU index/UUID or cpu')
 weights=a.weights.resolve()
 if not weights.is_file():p.error('Model file not found: '+str(weights))
 os.environ['CUDA_VISIBLE_DEVICES']='' if a.gpu=='cpu' else a.gpu
 os.environ['HOUSE_DEVICE']='cpu' if a.gpu=='cpu' else '0'
 os.environ['HOUSE_MODEL']=str(weights);os.environ['YOLO_AUTOINSTALL']='False'
 for key,path in [('YOLO_CONFIG_DIR',ROOT/'.cache/yolo'),('MPLCONFIGDIR',ROOT/'.cache/matplotlib'),('TMPDIR',ROOT/'.tmp')]:
  path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
 sys.path.insert(0,str(ROOT/'vendor'));sys.path.insert(0,str(ROOT/'code'))
 import torch
 if a.gpu!='cpu' and not torch.cuda.is_available():raise SystemExit('Requested GPU unavailable; check NVIDIA driver / CUDA PyTorch / GPU ID. Use --gpu cpu explicitly for CPU.')
 from disease_service.unique_app import create_app
 import uvicorn
 print(f'Physical GPU={a.gpu}; model={weights}; outputs={ROOT / "outputs"}',flush=True)
 uvicorn.run(create_app(),host=a.host,port=a.port,workers=1)
if __name__=='__main__':main()
