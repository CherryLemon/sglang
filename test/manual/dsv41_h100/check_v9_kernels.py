"""Exercise the actual integrated kernels, fixed-address graph replay and padding."""
import importlib.util
import json
import os
from pathlib import Path
import torch

os.environ['SGLANG_OPT_DSV41_INDEXER_SKIP_INVALID_TILES'] = '1'
os.environ['SGLANG_OPT_DSV41_DEEPSELECT_CANDIDATE_TOPK'] = '0'
from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import (
    fp4_index_logits_req_to_token, fp4_index_logits_candidate_blocks,
    unpack_fp4_index_keys_to_fp8, quantize_bf16_index_queries_fp8,
    fp8_index_logits_prefill,
)
from sglang.kernels.ops.attention.dsv4.candidate_blocks import candidate_block_state, finalize_candidate_topk

p = Path(__file__).with_name('reference_fp4_v8.py')
spec=importlib.util.spec_from_file_location('reference_fp4',p)
ref=importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
torch.manual_seed(91)
checks=0
for ratio in [1,2]:
 for width in [65,513,16385]:
  b,h,page=6,32,64
  alloc=((width+page-1)//page)*page
  q=torch.randint(-3,4,(b,h,128),device='cuda').to(torch.bfloat16)
  w=torch.rand(b,h,device='cuda',dtype=torch.bfloat16)
  table=torch.randint(0,256,(alloc//page,page*68),device='cuda',dtype=torch.uint8)
  table[:,page*64:]=127
  slots=torch.randperm(alloc,device='cuda')[:width].expand(b,-1).contiguous()
  req=torch.tensor([2,0,2,1,0,1],device='cuda',dtype=torch.int64)
  mapping=torch.zeros(3,width*ratio,device='cuda',dtype=torch.int32)
  mapping[:,::ratio]=slots[0]*ratio
  lens=torch.tensor([0,1,63,64,width-1,width],device='cuda',dtype=torch.int64)
  expected=ref.fp4_index_logits_decode(q,w,slots,lens,table,page,skip_invalid=False)
  actual=fp4_index_logits_req_to_token(q,w,mapping,req,lens,table,page,ratio,width)
  torch.testing.assert_close(actual,expected,rtol=0,atol=0)
  # Fused scores-only output must initialize all invisible tiles and row-zero length.
  bs,bl=fp4_index_logits_req_to_token(q,w,mapping,req,lens,table,page,ratio,width,candidate_block_size=8,write_logits=False)
  padded=torch.nn.functional.pad(expected,(0,(-width)%8),value=-torch.inf)
  want=padded.reshape(b,-1,8).amax(-1)
  for row in range(b):
   if lens[row]>0:want[row,(lens[row]-1)//8]=torch.inf
  torch.testing.assert_close(bs,want,rtol=0,atol=0)
  assert torch.equal(bl,(lens+7)//8)
  for _ in range(2):fp4_index_logits_req_to_token(q,w,mapping,req,lens,table,page,ratio,width)
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=fp4_index_logits_req_to_token(q,w,mapping,req,lens,table,page,ratio,width)
  for length in [width,0,64,1,width]:
   lens.fill_(length); g.replay()
   torch.testing.assert_close(out,ref.fp4_index_logits_decode(q,w,slots,lens,table,page,skip_invalid=False),rtol=0,atol=0)
   checks+=1
  # FP8 prefill keeps exact FP4-representable inputs in this controlled case.
  keys=unpack_fp4_index_keys_to_fp8(slots[0],table,page)
  pf=fp8_index_logits_prefill(quantize_bf16_index_queries_fp8(q),w,keys,lens)
  torch.testing.assert_close(pf[:,:width],out,rtol=0,atol=0)
  checks+=3
  if width>=16384:
   blocks,counts,plan=candidate_block_state(bs,bl,torch.tensor([0,1,63,64,width-1,width],device='cuda'),topk_blocks=2048,block_size=8)
   scores=fp4_index_logits_candidate_blocks(q,w,mapping,req,blocks,counts,table,page,ratio,8)
   logical=(blocks[:,:,None]*8+torch.arange(8,device='cuda')).reshape(b,-1).long()
   valid=torch.arange(logical.shape[1],device='cuda')[None,:]<counts[:,None]
   direct=expected.gather(1,logical.clamp(0,width-1)).masked_fill(~valid,-torch.inf)
   torch.testing.assert_close(scores,direct,rtol=0,atol=0)
   sel=scores.topk(512,-1).indices
   pages=torch.empty(b,512,device='cuda',dtype=torch.int32);raw=torch.empty_like(pages)
   finalize_candidate_topk(sel,scores,counts,mapping,req,pages,raw,ratio=ratio,candidate_blocks=blocks,candidate_block_size=8)
   assert bool(((pages>=0)|(pages==-1)).all())
   checks+=2
print(json.dumps({'passed':True,'checks':checks,'gpu':torch.cuda.get_device_name(0)}))
