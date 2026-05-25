
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
        add_31 = torch.ops.aten.add.Tensor(mul_31, arg0_1);  mul_31 = None
        mul_32 = torch.ops.aten.mul.Tensor(add_31, arg0_1);  add_31 = None
        add_32 = torch.ops.aten.add.Tensor(mul_32, arg0_1);  mul_32 = None
        mul_33 = torch.ops.aten.mul.Tensor(add_32, arg0_1);  add_32 = None
        add_33 = torch.ops.aten.add.Tensor(mul_33, arg0_1);  mul_33 = None
        mul_34 = torch.ops.aten.mul.Tensor(add_33, arg0_1);  add_33 = None
        add_34 = torch.ops.aten.add.Tensor(mul_34, arg0_1);  mul_34 = None
        mul_35 = torch.ops.aten.mul.Tensor(add_34, arg0_1);  add_34 = None
        add_35 = torch.ops.aten.add.Tensor(mul_35, arg0_1);  mul_35 = None
        mul_36 = torch.ops.aten.mul.Tensor(add_35, arg0_1);  add_35 = None
        add_36 = torch.ops.aten.add.Tensor(mul_36, arg0_1);  mul_36 = None
        mul_37 = torch.ops.aten.mul.Tensor(add_36, arg0_1);  add_36 = None
        add_37 = torch.ops.aten.add.Tensor(mul_37, arg0_1);  mul_37 = None
        mul_38 = torch.ops.aten.mul.Tensor(add_37, arg0_1);  add_37 = None
        add_38 = torch.ops.aten.add.Tensor(mul_38, arg0_1);  mul_38 = None
        mul_39 = torch.ops.aten.mul.Tensor(add_38, arg0_1);  add_38 = None
        add_39 = torch.ops.aten.add.Tensor(mul_39, arg0_1);  mul_39 = None
        mul_40 = torch.ops.aten.mul.Tensor(add_39, arg0_1);  add_39 = None
        add_40 = torch.ops.aten.add.Tensor(mul_40, arg0_1);  mul_40 = None
        mul_41 = torch.ops.aten.mul.Tensor(add_40, arg0_1);  add_40 = None
        add_41 = torch.ops.aten.add.Tensor(mul_41, arg0_1);  mul_41 = None
        mul_42 = torch.ops.aten.mul.Tensor(add_41, arg0_1);  add_41 = None
        add_42 = torch.ops.aten.add.Tensor(mul_42, arg0_1);  mul_42 = None
        mul_43 = torch.ops.aten.mul.Tensor(add_42, arg0_1);  add_42 = None
        add_43 = torch.ops.aten.add.Tensor(mul_43, arg0_1);  mul_43 = None
        mul_44 = torch.ops.aten.mul.Tensor(add_43, arg0_1);  add_43 = None
        add_44 = torch.ops.aten.add.Tensor(mul_44, arg0_1);  mul_44 = None
        mul_45 = torch.ops.aten.mul.Tensor(add_44, arg0_1);  add_44 = None
        add_45 = torch.ops.aten.add.Tensor(mul_45, arg0_1);  mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(add_45, arg0_1);  add_45 = None
        add_46 = torch.ops.aten.add.Tensor(mul_46, arg0_1);  mul_46 = None
        mul_47 = torch.ops.aten.mul.Tensor(add_46, arg0_1);  add_46 = None
        add_47 = torch.ops.aten.add.Tensor(mul_47, arg0_1);  mul_47 = None
        mul_48 = torch.ops.aten.mul.Tensor(add_47, arg0_1);  add_47 = None
        add_48 = torch.ops.aten.add.Tensor(mul_48, arg0_1);  mul_48 = None
        mul_49 = torch.ops.aten.mul.Tensor(add_48, arg0_1);  add_48 = None
        add_49 = torch.ops.aten.add.Tensor(mul_49, arg0_1);  mul_49 = None
        mul_50 = torch.ops.aten.mul.Tensor(add_49, arg0_1);  add_49 = None
        add_50 = torch.ops.aten.add.Tensor(mul_50, arg0_1);  mul_50 = None
        mul_51 = torch.ops.aten.mul.Tensor(add_50, arg0_1);  add_50 = None
        add_51 = torch.ops.aten.add.Tensor(mul_51, arg0_1);  mul_51 = None
        mul_52 = torch.ops.aten.mul.Tensor(add_51, arg0_1);  add_51 = None
        add_52 = torch.ops.aten.add.Tensor(mul_52, arg0_1);  mul_52 = None
        mul_53 = torch.ops.aten.mul.Tensor(add_52, arg0_1);  add_52 = None
        add_53 = torch.ops.aten.add.Tensor(mul_53, arg0_1);  mul_53 = None
        mul_54 = torch.ops.aten.mul.Tensor(add_53, arg0_1);  add_53 = None
        add_54 = torch.ops.aten.add.Tensor(mul_54, arg0_1);  mul_54 = None
        mul_55 = torch.ops.aten.mul.Tensor(add_54, arg0_1);  add_54 = None
        add_55 = torch.ops.aten.add.Tensor(mul_55, arg0_1);  mul_55 = None
        mul_56 = torch.ops.aten.mul.Tensor(add_55, arg0_1);  add_55 = None
        add_56 = torch.ops.aten.add.Tensor(mul_56, arg0_1);  mul_56 = None
        mul_57 = torch.ops.aten.mul.Tensor(add_56, arg0_1);  add_56 = None
        add_57 = torch.ops.aten.add.Tensor(mul_57, arg0_1);  mul_57 = None
        mul_58 = torch.ops.aten.mul.Tensor(add_57, arg0_1);  add_57 = None
        add_58 = torch.ops.aten.add.Tensor(mul_58, arg0_1);  mul_58 = None
        mul_59 = torch.ops.aten.mul.Tensor(add_58, arg0_1);  add_58 = None
        add_59 = torch.ops.aten.add.Tensor(mul_59, arg0_1);  mul_59 = None
        mul_60 = torch.ops.aten.mul.Tensor(add_59, arg0_1);  add_59 = None
        add_60 = torch.ops.aten.add.Tensor(mul_60, arg0_1);  mul_60 = None
        mul_61 = torch.ops.aten.mul.Tensor(add_60, arg0_1);  add_60 = None
        add_61 = torch.ops.aten.add.Tensor(mul_61, arg0_1);  mul_61 = None
        mul_62 = torch.ops.aten.mul.Tensor(add_61, arg0_1);  add_61 = None
        add_62 = torch.ops.aten.add.Tensor(mul_62, arg0_1);  mul_62 = None
        mul_63 = torch.ops.aten.mul.Tensor(add_62, arg0_1);  add_62 = None
        add_63 = torch.ops.aten.add.Tensor(mul_63, arg0_1);  mul_63 = None
        mul_64 = torch.ops.aten.mul.Tensor(add_63, arg0_1);  add_63 = None
        add_64 = torch.ops.aten.add.Tensor(mul_64, arg0_1);  mul_64 = None
        mul_65 = torch.ops.aten.mul.Tensor(add_64, arg0_1);  add_64 = None
        add_65 = torch.ops.aten.add.Tensor(mul_65, arg0_1);  mul_65 = None
        mul_66 = torch.ops.aten.mul.Tensor(add_65, arg0_1);  add_65 = None
        add_66 = torch.ops.aten.add.Tensor(mul_66, arg0_1);  mul_66 = None
        mul_67 = torch.ops.aten.mul.Tensor(add_66, arg0_1);  add_66 = None
        add_67 = torch.ops.aten.add.Tensor(mul_67, arg0_1);  mul_67 = None
        mul_68 = torch.ops.aten.mul.Tensor(add_67, arg0_1);  add_67 = None
        add_68 = torch.ops.aten.add.Tensor(mul_68, arg0_1);  mul_68 = None
        mul_69 = torch.ops.aten.mul.Tensor(add_68, arg0_1);  add_68 = None
        add_69 = torch.ops.aten.add.Tensor(mul_69, arg0_1);  mul_69 = None
        mul_70 = torch.ops.aten.mul.Tensor(add_69, arg0_1);  add_69 = None
        add_70 = torch.ops.aten.add.Tensor(mul_70, arg0_1);  mul_70 = None
        mul_71 = torch.ops.aten.mul.Tensor(add_70, arg0_1);  add_70 = None
        add_71 = torch.ops.aten.add.Tensor(mul_71, arg0_1);  mul_71 = None
        mul_72 = torch.ops.aten.mul.Tensor(add_71, arg0_1);  add_71 = None
        add_72 = torch.ops.aten.add.Tensor(mul_72, arg0_1);  mul_72 = None
        mul_73 = torch.ops.aten.mul.Tensor(add_72, arg0_1);  add_72 = None
        add_73 = torch.ops.aten.add.Tensor(mul_73, arg0_1);  mul_73 = None
        mul_74 = torch.ops.aten.mul.Tensor(add_73, arg0_1);  add_73 = None
        add_74 = torch.ops.aten.add.Tensor(mul_74, arg0_1);  mul_74 = None
        mul_75 = torch.ops.aten.mul.Tensor(add_74, arg0_1);  add_74 = None
        add_75 = torch.ops.aten.add.Tensor(mul_75, arg0_1);  mul_75 = None
        mul_76 = torch.ops.aten.mul.Tensor(add_75, arg0_1);  add_75 = None
        add_76 = torch.ops.aten.add.Tensor(mul_76, arg0_1);  mul_76 = None
        mul_77 = torch.ops.aten.mul.Tensor(add_76, arg0_1);  add_76 = None
        add_77 = torch.ops.aten.add.Tensor(mul_77, arg0_1);  mul_77 = None
        mul_78 = torch.ops.aten.mul.Tensor(add_77, arg0_1);  add_77 = None
        add_78 = torch.ops.aten.add.Tensor(mul_78, arg0_1);  mul_78 = None
        mul_79 = torch.ops.aten.mul.Tensor(add_78, arg0_1);  add_78 = None
        add_79 = torch.ops.aten.add.Tensor(mul_79, arg0_1);  mul_79 = None
        mul_80 = torch.ops.aten.mul.Tensor(add_79, arg0_1);  add_79 = None
        add_80 = torch.ops.aten.add.Tensor(mul_80, arg0_1);  mul_80 = None
        mul_81 = torch.ops.aten.mul.Tensor(add_80, arg0_1);  add_80 = None
        add_81 = torch.ops.aten.add.Tensor(mul_81, arg0_1);  mul_81 = None
        mul_82 = torch.ops.aten.mul.Tensor(add_81, arg0_1);  add_81 = None
        add_82 = torch.ops.aten.add.Tensor(mul_82, arg0_1);  mul_82 = None
        mul_83 = torch.ops.aten.mul.Tensor(add_82, arg0_1);  add_82 = None
        add_83 = torch.ops.aten.add.Tensor(mul_83, arg0_1);  mul_83 = None
        mul_84 = torch.ops.aten.mul.Tensor(add_83, arg0_1);  add_83 = None
        add_84 = torch.ops.aten.add.Tensor(mul_84, arg0_1);  mul_84 = None
        mul_85 = torch.ops.aten.mul.Tensor(add_84, arg0_1);  add_84 = None
        add_85 = torch.ops.aten.add.Tensor(mul_85, arg0_1);  mul_85 = None
        mul_86 = torch.ops.aten.mul.Tensor(add_85, arg0_1);  add_85 = None
        add_86 = torch.ops.aten.add.Tensor(mul_86, arg0_1);  mul_86 = None
        mul_87 = torch.ops.aten.mul.Tensor(add_86, arg0_1);  add_86 = None
        add_87 = torch.ops.aten.add.Tensor(mul_87, arg0_1);  mul_87 = None
        mul_88 = torch.ops.aten.mul.Tensor(add_87, arg0_1);  add_87 = None
        add_88 = torch.ops.aten.add.Tensor(mul_88, arg0_1);  mul_88 = None
        mul_89 = torch.ops.aten.mul.Tensor(add_88, arg0_1);  add_88 = None
        add_89 = torch.ops.aten.add.Tensor(mul_89, arg0_1);  mul_89 = None
        mul_90 = torch.ops.aten.mul.Tensor(add_89, arg0_1);  add_89 = None
        add_90 = torch.ops.aten.add.Tensor(mul_90, arg0_1);  mul_90 = None
        mul_91 = torch.ops.aten.mul.Tensor(add_90, arg0_1);  add_90 = None
        add_91 = torch.ops.aten.add.Tensor(mul_91, arg0_1);  mul_91 = None
        mul_92 = torch.ops.aten.mul.Tensor(add_91, arg0_1);  add_91 = None
        add_92 = torch.ops.aten.add.Tensor(mul_92, arg0_1);  mul_92 = None
        mul_93 = torch.ops.aten.mul.Tensor(add_92, arg0_1);  add_92 = None
        add_93 = torch.ops.aten.add.Tensor(mul_93, arg0_1);  mul_93 = None
        mul_94 = torch.ops.aten.mul.Tensor(add_93, arg0_1);  add_93 = None
        add_94 = torch.ops.aten.add.Tensor(mul_94, arg0_1);  mul_94 = None
        mul_95 = torch.ops.aten.mul.Tensor(add_94, arg0_1);  add_94 = None
        add_95 = torch.ops.aten.add.Tensor(mul_95, arg0_1);  mul_95 = None
        mul_96 = torch.ops.aten.mul.Tensor(add_95, arg0_1);  add_95 = None
        add_96 = torch.ops.aten.add.Tensor(mul_96, arg0_1);  mul_96 = None
        mul_97 = torch.ops.aten.mul.Tensor(add_96, arg0_1);  add_96 = None
        add_97 = torch.ops.aten.add.Tensor(mul_97, arg0_1);  mul_97 = None
        mul_98 = torch.ops.aten.mul.Tensor(add_97, arg0_1);  add_97 = None
        add_98 = torch.ops.aten.add.Tensor(mul_98, arg0_1);  mul_98 = None
        mul_99 = torch.ops.aten.mul.Tensor(add_98, arg0_1);  add_98 = None
        add_99 = torch.ops.aten.add.Tensor(mul_99, arg0_1);  mul_99 = None
        mul_100 = torch.ops.aten.mul.Tensor(add_99, arg0_1);  add_99 = None
        add_100 = torch.ops.aten.add.Tensor(mul_100, arg0_1);  mul_100 = None
        mul_101 = torch.ops.aten.mul.Tensor(add_100, arg0_1);  add_100 = None
        add_101 = torch.ops.aten.add.Tensor(mul_101, arg0_1);  mul_101 = None
        mul_102 = torch.ops.aten.mul.Tensor(add_101, arg0_1);  add_101 = None
        add_102 = torch.ops.aten.add.Tensor(mul_102, arg0_1);  mul_102 = None
        mul_103 = torch.ops.aten.mul.Tensor(add_102, arg0_1);  add_102 = None
        add_103 = torch.ops.aten.add.Tensor(mul_103, arg0_1);  mul_103 = None
        mul_104 = torch.ops.aten.mul.Tensor(add_103, arg0_1);  add_103 = None
        add_104 = torch.ops.aten.add.Tensor(mul_104, arg0_1);  mul_104 = None
        mul_105 = torch.ops.aten.mul.Tensor(add_104, arg0_1);  add_104 = None
        add_105 = torch.ops.aten.add.Tensor(mul_105, arg0_1);  mul_105 = None
        mul_106 = torch.ops.aten.mul.Tensor(add_105, arg0_1);  add_105 = None
        add_106 = torch.ops.aten.add.Tensor(mul_106, arg0_1);  mul_106 = None
        mul_107 = torch.ops.aten.mul.Tensor(add_106, arg0_1);  add_106 = None
        add_107 = torch.ops.aten.add.Tensor(mul_107, arg0_1);  mul_107 = None
        mul_108 = torch.ops.aten.mul.Tensor(add_107, arg0_1);  add_107 = None
        add_108 = torch.ops.aten.add.Tensor(mul_108, arg0_1);  mul_108 = None
        mul_109 = torch.ops.aten.mul.Tensor(add_108, arg0_1);  add_108 = None
        add_109 = torch.ops.aten.add.Tensor(mul_109, arg0_1);  mul_109 = None
        mul_110 = torch.ops.aten.mul.Tensor(add_109, arg0_1);  add_109 = None
        add_110 = torch.ops.aten.add.Tensor(mul_110, arg0_1);  mul_110 = None
        mul_111 = torch.ops.aten.mul.Tensor(add_110, arg0_1);  add_110 = None
        add_111 = torch.ops.aten.add.Tensor(mul_111, arg0_1);  mul_111 = None
        mul_112 = torch.ops.aten.mul.Tensor(add_111, arg0_1);  add_111 = None
        add_112 = torch.ops.aten.add.Tensor(mul_112, arg0_1);  mul_112 = None
        mul_113 = torch.ops.aten.mul.Tensor(add_112, arg0_1);  add_112 = None
        add_113 = torch.ops.aten.add.Tensor(mul_113, arg0_1);  mul_113 = None
        mul_114 = torch.ops.aten.mul.Tensor(add_113, arg0_1);  add_113 = None
        add_114 = torch.ops.aten.add.Tensor(mul_114, arg0_1);  mul_114 = None
        mul_115 = torch.ops.aten.mul.Tensor(add_114, arg0_1);  add_114 = None
        add_115 = torch.ops.aten.add.Tensor(mul_115, arg0_1);  mul_115 = None
        mul_116 = torch.ops.aten.mul.Tensor(add_115, arg0_1);  add_115 = None
        add_116 = torch.ops.aten.add.Tensor(mul_116, arg0_1);  mul_116 = None
        mul_117 = torch.ops.aten.mul.Tensor(add_116, arg0_1);  add_116 = None
        add_117 = torch.ops.aten.add.Tensor(mul_117, arg0_1);  mul_117 = None
        mul_118 = torch.ops.aten.mul.Tensor(add_117, arg0_1);  add_117 = None
        add_118 = torch.ops.aten.add.Tensor(mul_118, arg0_1);  mul_118 = None
        mul_119 = torch.ops.aten.mul.Tensor(add_118, arg0_1);  add_118 = None
        add_119 = torch.ops.aten.add.Tensor(mul_119, arg0_1);  mul_119 = None
        mul_120 = torch.ops.aten.mul.Tensor(add_119, arg0_1);  add_119 = None
        add_120 = torch.ops.aten.add.Tensor(mul_120, arg0_1);  mul_120 = None
        mul_121 = torch.ops.aten.mul.Tensor(add_120, arg0_1);  add_120 = None
        add_121 = torch.ops.aten.add.Tensor(mul_121, arg0_1);  mul_121 = None
        mul_122 = torch.ops.aten.mul.Tensor(add_121, arg0_1);  add_121 = None
        add_122 = torch.ops.aten.add.Tensor(mul_122, arg0_1);  mul_122 = None
        mul_123 = torch.ops.aten.mul.Tensor(add_122, arg0_1);  add_122 = None
        add_123 = torch.ops.aten.add.Tensor(mul_123, arg0_1);  mul_123 = None
        mul_124 = torch.ops.aten.mul.Tensor(add_123, arg0_1);  add_123 = None
        add_124 = torch.ops.aten.add.Tensor(mul_124, arg0_1);  mul_124 = None
        mul_125 = torch.ops.aten.mul.Tensor(add_124, arg0_1);  add_124 = None
        add_125 = torch.ops.aten.add.Tensor(mul_125, arg0_1);  mul_125 = None
        mul_126 = torch.ops.aten.mul.Tensor(add_125, arg0_1);  add_125 = None
        add_126 = torch.ops.aten.add.Tensor(mul_126, arg0_1);  mul_126 = None
        mul_127 = torch.ops.aten.mul.Tensor(add_126, arg0_1);  add_126 = None
        add_127 = torch.ops.aten.add.Tensor(mul_127, arg0_1);  mul_127 = arg0_1 = None
        return (add_127,)
        
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