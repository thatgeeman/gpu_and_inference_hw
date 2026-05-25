
import os
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_COMPILE_LOG_LEVEL'] = 'debug'
os.environ['TORCHDYNAMO_VERBOSE'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_gg'
os.environ['TRITON_CACHE_DIR'] = '/tmp/torchinductor_gg/triton/0'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None




# torch version: 2.9.1+cu128
# torch cuda version: 12.8
# torch git version: 5811a8d7da873dd699ff6687092c225caffcf1bb


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Tue_Oct_29_23:50:19_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.85 
# Build cuda_12.6.r12.6/compiler.35059454_0 

# GPU Hardware Info: 
# NVIDIA RTX 2000 Ada Generation Laptop GPU : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1):
        mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1)
        add = torch.ops.aten.add.Tensor(mul, arg0_1);  mul = None
        mul_1 = torch.ops.aten.mul.Tensor(add, arg0_1);  add = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, arg0_1);  mul_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(add_1, arg0_1);  add_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_2, arg0_1);  mul_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(add_2, arg0_1);  add_2 = None
        add_3 = torch.ops.aten.add.Tensor(mul_3, arg0_1);  mul_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(add_3, arg0_1);  add_3 = None
        add_4 = torch.ops.aten.add.Tensor(mul_4, arg0_1);  mul_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(add_4, arg0_1);  add_4 = None
        add_5 = torch.ops.aten.add.Tensor(mul_5, arg0_1);  mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(add_5, arg0_1);  add_5 = None
        add_6 = torch.ops.aten.add.Tensor(mul_6, arg0_1);  mul_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(add_6, arg0_1);  add_6 = None
        add_7 = torch.ops.aten.add.Tensor(mul_7, arg0_1);  mul_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(add_7, arg0_1);  add_7 = None
        add_8 = torch.ops.aten.add.Tensor(mul_8, arg0_1);  mul_8 = None
        mul_9 = torch.ops.aten.mul.Tensor(add_8, arg0_1);  add_8 = None
        add_9 = torch.ops.aten.add.Tensor(mul_9, arg0_1);  mul_9 = None
        mul_10 = torch.ops.aten.mul.Tensor(add_9, arg0_1);  add_9 = None
        add_10 = torch.ops.aten.add.Tensor(mul_10, arg0_1);  mul_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(add_10, arg0_1);  add_10 = None
        add_11 = torch.ops.aten.add.Tensor(mul_11, arg0_1);  mul_11 = None
        mul_12 = torch.ops.aten.mul.Tensor(add_11, arg0_1);  add_11 = None
        add_12 = torch.ops.aten.add.Tensor(mul_12, arg0_1);  mul_12 = None
        mul_13 = torch.ops.aten.mul.Tensor(add_12, arg0_1);  add_12 = None
        add_13 = torch.ops.aten.add.Tensor(mul_13, arg0_1);  mul_13 = None
        mul_14 = torch.ops.aten.mul.Tensor(add_13, arg0_1);  add_13 = None
        add_14 = torch.ops.aten.add.Tensor(mul_14, arg0_1);  mul_14 = None
        mul_15 = torch.ops.aten.mul.Tensor(add_14, arg0_1);  add_14 = None
        add_15 = torch.ops.aten.add.Tensor(mul_15, arg0_1);  mul_15 = None
        mul_16 = torch.ops.aten.mul.Tensor(add_15, arg0_1);  add_15 = None
        add_16 = torch.ops.aten.add.Tensor(mul_16, arg0_1);  mul_16 = None
        mul_17 = torch.ops.aten.mul.Tensor(add_16, arg0_1);  add_16 = None
        add_17 = torch.ops.aten.add.Tensor(mul_17, arg0_1);  mul_17 = None
        mul_18 = torch.ops.aten.mul.Tensor(add_17, arg0_1);  add_17 = None
        add_18 = torch.ops.aten.add.Tensor(mul_18, arg0_1);  mul_18 = None
        mul_19 = torch.ops.aten.mul.Tensor(add_18, arg0_1);  add_18 = None
        add_19 = torch.ops.aten.add.Tensor(mul_19, arg0_1);  mul_19 = None
        mul_20 = torch.ops.aten.mul.Tensor(add_19, arg0_1);  add_19 = None
        add_20 = torch.ops.aten.add.Tensor(mul_20, arg0_1);  mul_20 = None
        mul_21 = torch.ops.aten.mul.Tensor(add_20, arg0_1);  add_20 = None
        add_21 = torch.ops.aten.add.Tensor(mul_21, arg0_1);  mul_21 = None
        mul_22 = torch.ops.aten.mul.Tensor(add_21, arg0_1);  add_21 = None
        add_22 = torch.ops.aten.add.Tensor(mul_22, arg0_1);  mul_22 = None
        mul_23 = torch.ops.aten.mul.Tensor(add_22, arg0_1);  add_22 = None
        add_23 = torch.ops.aten.add.Tensor(mul_23, arg0_1);  mul_23 = None
        mul_24 = torch.ops.aten.mul.Tensor(add_23, arg0_1);  add_23 = None
        add_24 = torch.ops.aten.add.Tensor(mul_24, arg0_1);  mul_24 = None
        mul_25 = torch.ops.aten.mul.Tensor(add_24, arg0_1);  add_24 = None
        add_25 = torch.ops.aten.add.Tensor(mul_25, arg0_1);  mul_25 = None
        mul_26 = torch.ops.aten.mul.Tensor(add_25, arg0_1);  add_25 = None
        add_26 = torch.ops.aten.add.Tensor(mul_26, arg0_1);  mul_26 = None
        mul_27 = torch.ops.aten.mul.Tensor(add_26, arg0_1);  add_26 = None
        add_27 = torch.ops.aten.add.Tensor(mul_27, arg0_1);  mul_27 = None
        mul_28 = torch.ops.aten.mul.Tensor(add_27, arg0_1);  add_27 = None
        add_28 = torch.ops.aten.add.Tensor(mul_28, arg0_1);  mul_28 = None
        mul_29 = torch.ops.aten.mul.Tensor(add_28, arg0_1);  add_28 = None
        add_29 = torch.ops.aten.add.Tensor(mul_29, arg0_1);  mul_29 = None
        mul_30 = torch.ops.aten.mul.Tensor(add_29, arg0_1);  add_29 = None
        add_30 = torch.ops.aten.add.Tensor(mul_30, arg0_1);  mul_30 = None
        mul_31 = torch.ops.aten.mul.Tensor(add_30, arg0_1);  add_30 = None
        add_31 = torch.ops.aten.add.Tensor(mul_31, arg0_1);  mul_31 = arg0_1 = None
        return (add_31,)
        
def load_args(reader):
    buf0 = reader.storage(None, 268435456, device=device(type='cuda', index=0))
    reader.tensor(buf0, (67108864,), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)