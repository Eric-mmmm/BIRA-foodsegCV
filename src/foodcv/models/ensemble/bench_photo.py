"""Benchmark photo RGB, TensorRT 10, formes fixes et batch 1."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from torchvision.transforms import functional as T

ROOT = Path(__file__).resolve().parent
LOGGER = trt.Logger(trt.Logger.WARNING)

class Engine:
    def __init__(self, runtime, name):
        self.engine = runtime.deserialize_cuda_engine((ROOT/'engines'/f'{name}.plan').read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Moteur illisible: {name}')
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input, = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output, = [n for n in names if n != self.input]
        self.y = torch.empty(tuple(self.engine.get_tensor_shape(self.output)), device='cuda',
                             dtype=torch.from_numpy(np.empty((),trt.nptype(self.engine.get_tensor_dtype(self.output)))).dtype)

    def __call__(self,x):
        self.ctx.set_tensor_address(self.input,x.data_ptr())
        self.ctx.set_tensor_address(self.output,self.y.data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        return self.y.float()

def tensor(image, size, crop=None):
    image=T.resize(image,size)
    if crop: image=T.center_crop(image,crop)
    return T.normalize(T.to_tensor(image),[.485,.456,.406],[.229,.224,.225]).unsqueeze(0).cuda().contiguous()

def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('photo')
    parser.add_argument('--dish',choices=['efficientnet','convnext'],default='efficientnet')
    parser.add_argument('--runs',type=int,default=100)
    a=parser.parse_args()
    if a.runs<1: parser.error('--runs doit être positif')
    spec=json.loads((ROOT/(a.dish+'.json')).read_text())['dish']
    region=json.loads((ROOT/'deeplab.json').read_text())['region']
    guard=json.loads((ROOT/'b1.json').read_text())
    with Image.open(a.photo) as f: image=f.convert('RGB')
    runtime=trt.Runtime(LOGGER)
    b1,dish,seg=[Engine(runtime,n) for n in ['b1',a.dish,'deeplab']]
    times=[]
    for i in range(5+a.runs):
        t=time.perf_counter()
        food=b1(tensor(image,256,224)).softmax(1)[0,1].item()
        probs=dish(tensor(image,365,320)).softmax(1)[0]
        idx=int(probs.argmax().item())
        score=probs[idx].item()
        label=spec['classes'][idx]
        logits=seg(tensor(image,[640,640]))
        mask=torch.nn.functional.interpolate(logits,(image.height,image.width),mode='bilinear',align_corners=False).argmax(1)[0].byte().cpu().numpy()
        counts=np.bincount(mask.ravel(),minlength=9)
        for k in range(1,9):
            if counts[k]<mask.size*region['minimum_area_fraction']: mask[mask==k]=0
        if i>=5: times.append(1000*(time.perf_counter()-t))
    Image.fromarray(mask).save('classes.png')
    accepted=score>=spec.get('per_class_thresholds',{}).get(label,spec['acceptance_threshold'])
    print(json.dumps(dict(food_score=food,food_accepted=food>=guard['threshold'],dish=label,dish_accepted=accepted,
                          p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),runs=a.runs)))
if __name__=='__main__': run()
