# AOT ID: ['6_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_gg/lb/clbhsinnvjupdc4xuisa6cmlcro7qgd6wkrh7kh3ifmnvu5rrnsn.py
# Topologically Sorted Source Nodes: [mul, acc, mul_1, acc_1, mul_2, acc_2, mul_3, acc_3, mul_4, acc_4, mul_5, acc_5, mul_6, acc_6, mul_7, acc_7, mul_8, acc_8, mul_9, acc_9, mul_10, acc_10, mul_11, acc_11, mul_12, acc_12, mul_13, acc_13, mul_14, acc_14, mul_15, acc_15, mul_16, acc_16, mul_17, acc_17, mul_18, acc_18, mul_19, acc_19, mul_20, acc_20, mul_21, acc_21, mul_22, acc_22, mul_23, acc_23, mul_24, acc_24, mul_25, acc_25, mul_26, acc_26, mul_27, acc_27, mul_28, acc_28, mul_29, acc_29, mul_30, acc_30, mul_31, acc_31, mul_32, acc_32, mul_33, acc_33, mul_34, acc_34, mul_35, acc_35, mul_36, acc_36, mul_37, acc_37, mul_38, acc_38, mul_39, acc_39, mul_40, acc_40, mul_41, acc_41, mul_42, acc_42, mul_43, acc_43, mul_44, acc_44, mul_45, acc_45, mul_46, acc_46, mul_47, acc_47, mul_48, acc_48, mul_49, acc_49, mul_50, acc_50, mul_51, acc_51, mul_52, acc_52, mul_53, acc_53, mul_54, acc_54, mul_55, acc_55, mul_56, acc_56, mul_57, acc_57, mul_58, acc_58, mul_59, acc_59, mul_60, acc_60, mul_61, acc_61, mul_62, acc_62, mul_63, acc_63], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   acc => add
#   acc_1 => add_1
#   acc_10 => add_10
#   acc_11 => add_11
#   acc_12 => add_12
#   acc_13 => add_13
#   acc_14 => add_14
#   acc_15 => add_15
#   acc_16 => add_16
#   acc_17 => add_17
#   acc_18 => add_18
#   acc_19 => add_19
#   acc_2 => add_2
#   acc_20 => add_20
#   acc_21 => add_21
#   acc_22 => add_22
#   acc_23 => add_23
#   acc_24 => add_24
#   acc_25 => add_25
#   acc_26 => add_26
#   acc_27 => add_27
#   acc_28 => add_28
#   acc_29 => add_29
#   acc_3 => add_3
#   acc_30 => add_30
#   acc_31 => add_31
#   acc_32 => add_32
#   acc_33 => add_33
#   acc_34 => add_34
#   acc_35 => add_35
#   acc_36 => add_36
#   acc_37 => add_37
#   acc_38 => add_38
#   acc_39 => add_39
#   acc_4 => add_4
#   acc_40 => add_40
#   acc_41 => add_41
#   acc_42 => add_42
#   acc_43 => add_43
#   acc_44 => add_44
#   acc_45 => add_45
#   acc_46 => add_46
#   acc_47 => add_47
#   acc_48 => add_48
#   acc_49 => add_49
#   acc_5 => add_5
#   acc_50 => add_50
#   acc_51 => add_51
#   acc_52 => add_52
#   acc_53 => add_53
#   acc_54 => add_54
#   acc_55 => add_55
#   acc_56 => add_56
#   acc_57 => add_57
#   acc_58 => add_58
#   acc_59 => add_59
#   acc_6 => add_6
#   acc_60 => add_60
#   acc_61 => add_61
#   acc_62 => add_62
#   acc_63 => add_63
#   acc_7 => add_7
#   acc_8 => add_8
#   acc_9 => add_9
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_2 => mul_2
#   mul_20 => mul_20
#   mul_21 => mul_21
#   mul_22 => mul_22
#   mul_23 => mul_23
#   mul_24 => mul_24
#   mul_25 => mul_25
#   mul_26 => mul_26
#   mul_27 => mul_27
#   mul_28 => mul_28
#   mul_29 => mul_29
#   mul_3 => mul_3
#   mul_30 => mul_30
#   mul_31 => mul_31
#   mul_32 => mul_32
#   mul_33 => mul_33
#   mul_34 => mul_34
#   mul_35 => mul_35
#   mul_36 => mul_36
#   mul_37 => mul_37
#   mul_38 => mul_38
#   mul_39 => mul_39
#   mul_4 => mul_4
#   mul_40 => mul_40
#   mul_41 => mul_41
#   mul_42 => mul_42
#   mul_43 => mul_43
#   mul_44 => mul_44
#   mul_45 => mul_45
#   mul_46 => mul_46
#   mul_47 => mul_47
#   mul_48 => mul_48
#   mul_49 => mul_49
#   mul_5 => mul_5
#   mul_50 => mul_50
#   mul_51 => mul_51
#   mul_52 => mul_52
#   mul_53 => mul_53
#   mul_54 => mul_54
#   mul_55 => mul_55
#   mul_56 => mul_56
#   mul_57 => mul_57
#   mul_58 => mul_58
#   mul_59 => mul_59
#   mul_6 => mul_6
#   mul_60 => mul_60
#   mul_61 => mul_61
#   mul_62 => mul_62
#   mul_63 => mul_63
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
# Graph fragment:
#   %arg0_1 : Tensor "f32[67108864][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %add_49 : Tensor "f32[67108864][1]cuda:0" = PlaceHolder[target=add_49]
#   %mul : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg0_1), kwargs = {})
#   %add : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %arg0_1), kwargs = {})
#   %mul_1 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %arg0_1), kwargs = {})
#   %add_1 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg0_1), kwargs = {})
#   %mul_2 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %arg0_1), kwargs = {})
#   %add_2 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %arg0_1), kwargs = {})
#   %mul_3 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %arg0_1), kwargs = {})
#   %add_3 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %arg0_1), kwargs = {})
#   %mul_4 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %arg0_1), kwargs = {})
#   %add_4 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg0_1), kwargs = {})
#   %mul_5 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %arg0_1), kwargs = {})
#   %add_5 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %arg0_1), kwargs = {})
#   %mul_6 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %arg0_1), kwargs = {})
#   %add_6 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %arg0_1), kwargs = {})
#   %mul_7 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %arg0_1), kwargs = {})
#   %add_7 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %arg0_1), kwargs = {})
#   %mul_8 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, %arg0_1), kwargs = {})
#   %add_8 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %arg0_1), kwargs = {})
#   %mul_9 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %arg0_1), kwargs = {})
#   %add_9 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %arg0_1), kwargs = {})
#   %mul_10 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, %arg0_1), kwargs = {})
#   %add_10 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %arg0_1), kwargs = {})
#   %mul_11 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %arg0_1), kwargs = {})
#   %add_11 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %arg0_1), kwargs = {})
#   %mul_12 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %arg0_1), kwargs = {})
#   %add_12 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %arg0_1), kwargs = {})
#   %mul_13 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %arg0_1), kwargs = {})
#   %add_13 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %arg0_1), kwargs = {})
#   %mul_14 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %arg0_1), kwargs = {})
#   %add_14 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %arg0_1), kwargs = {})
#   %mul_15 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_14, %arg0_1), kwargs = {})
#   %add_15 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %arg0_1), kwargs = {})
#   %mul_16 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %arg0_1), kwargs = {})
#   %add_16 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %arg0_1), kwargs = {})
#   %mul_17 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %arg0_1), kwargs = {})
#   %add_17 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %arg0_1), kwargs = {})
#   %mul_18 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %arg0_1), kwargs = {})
#   %add_18 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %arg0_1), kwargs = {})
#   %mul_19 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %arg0_1), kwargs = {})
#   %add_19 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %arg0_1), kwargs = {})
#   %mul_20 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, %arg0_1), kwargs = {})
#   %add_20 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %arg0_1), kwargs = {})
#   %mul_21 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_20, %arg0_1), kwargs = {})
#   %add_21 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %arg0_1), kwargs = {})
#   %mul_22 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %arg0_1), kwargs = {})
#   %add_22 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %arg0_1), kwargs = {})
#   %mul_23 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %arg0_1), kwargs = {})
#   %add_23 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %arg0_1), kwargs = {})
#   %mul_24 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %arg0_1), kwargs = {})
#   %add_24 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %arg0_1), kwargs = {})
#   %mul_25 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %arg0_1), kwargs = {})
#   %add_25 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %arg0_1), kwargs = {})
#   %mul_26 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_25, %arg0_1), kwargs = {})
#   %add_26 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %arg0_1), kwargs = {})
#   %mul_27 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %arg0_1), kwargs = {})
#   %add_27 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %arg0_1), kwargs = {})
#   %mul_28 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %arg0_1), kwargs = {})
#   %add_28 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %arg0_1), kwargs = {})
#   %mul_29 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_28, %arg0_1), kwargs = {})
#   %add_29 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %arg0_1), kwargs = {})
#   %mul_30 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_29, %arg0_1), kwargs = {})
#   %add_30 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %arg0_1), kwargs = {})
#   %mul_31 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_30, %arg0_1), kwargs = {})
#   %add_31 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %arg0_1), kwargs = {})
#   %mul_32 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_31, %arg0_1), kwargs = {})
#   %add_32 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %arg0_1), kwargs = {})
#   %mul_33 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, %arg0_1), kwargs = {})
#   %add_33 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_33, %arg0_1), kwargs = {})
#   %mul_34 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_33, %arg0_1), kwargs = {})
#   %add_34 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %arg0_1), kwargs = {})
#   %mul_35 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, %arg0_1), kwargs = {})
#   %add_35 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %arg0_1), kwargs = {})
#   %mul_36 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_35, %arg0_1), kwargs = {})
#   %add_36 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %arg0_1), kwargs = {})
#   %mul_37 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_36, %arg0_1), kwargs = {})
#   %add_37 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %arg0_1), kwargs = {})
#   %mul_38 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %arg0_1), kwargs = {})
#   %add_38 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %arg0_1), kwargs = {})
#   %mul_39 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %arg0_1), kwargs = {})
#   %add_39 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, %arg0_1), kwargs = {})
#   %mul_40 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_39, %arg0_1), kwargs = {})
#   %add_40 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_40, %arg0_1), kwargs = {})
#   %mul_41 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_40, %arg0_1), kwargs = {})
#   %add_41 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %arg0_1), kwargs = {})
#   %mul_42 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_41, %arg0_1), kwargs = {})
#   %add_42 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %arg0_1), kwargs = {})
#   %mul_43 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_42, %arg0_1), kwargs = {})
#   %add_43 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %arg0_1), kwargs = {})
#   %mul_44 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %arg0_1), kwargs = {})
#   %add_44 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %arg0_1), kwargs = {})
#   %mul_45 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_44, %arg0_1), kwargs = {})
#   %add_45 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %arg0_1), kwargs = {})
#   %mul_46 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_45, %arg0_1), kwargs = {})
#   %add_46 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, %arg0_1), kwargs = {})
#   %mul_47 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_46, %arg0_1), kwargs = {})
#   %add_47 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %arg0_1), kwargs = {})
#   %mul_48 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %arg0_1), kwargs = {})
#   %add_48 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %arg0_1), kwargs = {})
#   %mul_49 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_48, %arg0_1), kwargs = {})
#   %add_49 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %arg0_1), kwargs = {})
#   %mul_50 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_49, %arg0_1), kwargs = {})
#   %add_50 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %arg0_1), kwargs = {})
#   %mul_51 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_50, %arg0_1), kwargs = {})
#   %add_51 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, %arg0_1), kwargs = {})
#   %mul_52 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, %arg0_1), kwargs = {})
#   %add_52 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_52, %arg0_1), kwargs = {})
#   %mul_53 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_52, %arg0_1), kwargs = {})
#   %add_53 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %arg0_1), kwargs = {})
#   %mul_54 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %arg0_1), kwargs = {})
#   %add_54 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_54, %arg0_1), kwargs = {})
#   %mul_55 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_54, %arg0_1), kwargs = {})
#   %add_55 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %arg0_1), kwargs = {})
#   %mul_56 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %arg0_1), kwargs = {})
#   %add_56 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %arg0_1), kwargs = {})
#   %mul_57 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_56, %arg0_1), kwargs = {})
#   %add_57 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_57, %arg0_1), kwargs = {})
#   %mul_58 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %arg0_1), kwargs = {})
#   %add_58 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_58, %arg0_1), kwargs = {})
#   %mul_59 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, %arg0_1), kwargs = {})
#   %add_59 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %arg0_1), kwargs = {})
#   %mul_60 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_59, %arg0_1), kwargs = {})
#   %add_60 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_60, %arg0_1), kwargs = {})
#   %mul_61 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_60, %arg0_1), kwargs = {})
#   %add_61 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_61, %arg0_1), kwargs = {})
#   %mul_62 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_61, %arg0_1), kwargs = {})
#   %add_62 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %arg0_1), kwargs = {})
#   %mul_63 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_62, %arg0_1), kwargs = {})
#   %add_63 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_63, %arg0_1), kwargs = {})
#   return %add_49,%add_63
triton_poi_fused_add_mul_0 = async_compile.triton('triton_poi_fused_add_mul_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=24, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6E2CBEB9BFB65B06FFC57A5E69F68D54F759F46CC4C44428A4C003EBC7ADE616', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 805306368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0 * tmp0
    tmp2 = tmp1 + tmp0
    tmp3 = tmp2 * tmp0
    tmp4 = tmp3 + tmp0
    tmp5 = tmp4 * tmp0
    tmp6 = tmp5 + tmp0
    tmp7 = tmp6 * tmp0
    tmp8 = tmp7 + tmp0
    tmp9 = tmp8 * tmp0
    tmp10 = tmp9 + tmp0
    tmp11 = tmp10 * tmp0
    tmp12 = tmp11 + tmp0
    tmp13 = tmp12 * tmp0
    tmp14 = tmp13 + tmp0
    tmp15 = tmp14 * tmp0
    tmp16 = tmp15 + tmp0
    tmp17 = tmp16 * tmp0
    tmp18 = tmp17 + tmp0
    tmp19 = tmp18 * tmp0
    tmp20 = tmp19 + tmp0
    tmp21 = tmp20 * tmp0
    tmp22 = tmp21 + tmp0
    tmp23 = tmp22 * tmp0
    tmp24 = tmp23 + tmp0
    tmp25 = tmp24 * tmp0
    tmp26 = tmp25 + tmp0
    tmp27 = tmp26 * tmp0
    tmp28 = tmp27 + tmp0
    tmp29 = tmp28 * tmp0
    tmp30 = tmp29 + tmp0
    tmp31 = tmp30 * tmp0
    tmp32 = tmp31 + tmp0
    tmp33 = tmp32 * tmp0
    tmp34 = tmp33 + tmp0
    tmp35 = tmp34 * tmp0
    tmp36 = tmp35 + tmp0
    tmp37 = tmp36 * tmp0
    tmp38 = tmp37 + tmp0
    tmp39 = tmp38 * tmp0
    tmp40 = tmp39 + tmp0
    tmp41 = tmp40 * tmp0
    tmp42 = tmp41 + tmp0
    tmp43 = tmp42 * tmp0
    tmp44 = tmp43 + tmp0
    tmp45 = tmp44 * tmp0
    tmp46 = tmp45 + tmp0
    tmp47 = tmp46 * tmp0
    tmp48 = tmp47 + tmp0
    tmp49 = tmp48 * tmp0
    tmp50 = tmp49 + tmp0
    tmp51 = tmp50 * tmp0
    tmp52 = tmp51 + tmp0
    tmp53 = tmp52 * tmp0
    tmp54 = tmp53 + tmp0
    tmp55 = tmp54 * tmp0
    tmp56 = tmp55 + tmp0
    tmp57 = tmp56 * tmp0
    tmp58 = tmp57 + tmp0
    tmp59 = tmp58 * tmp0
    tmp60 = tmp59 + tmp0
    tmp61 = tmp60 * tmp0
    tmp62 = tmp61 + tmp0
    tmp63 = tmp62 * tmp0
    tmp64 = tmp63 + tmp0
    tmp65 = tmp64 * tmp0
    tmp66 = tmp65 + tmp0
    tmp67 = tmp66 * tmp0
    tmp68 = tmp67 + tmp0
    tmp69 = tmp68 * tmp0
    tmp70 = tmp69 + tmp0
    tmp71 = tmp70 * tmp0
    tmp72 = tmp71 + tmp0
    tmp73 = tmp72 * tmp0
    tmp74 = tmp73 + tmp0
    tmp75 = tmp74 * tmp0
    tmp76 = tmp75 + tmp0
    tmp77 = tmp76 * tmp0
    tmp78 = tmp77 + tmp0
    tmp79 = tmp78 * tmp0
    tmp80 = tmp79 + tmp0
    tmp81 = tmp80 * tmp0
    tmp82 = tmp81 + tmp0
    tmp83 = tmp82 * tmp0
    tmp84 = tmp83 + tmp0
    tmp85 = tmp84 * tmp0
    tmp86 = tmp85 + tmp0
    tmp87 = tmp86 * tmp0
    tmp88 = tmp87 + tmp0
    tmp89 = tmp88 * tmp0
    tmp90 = tmp89 + tmp0
    tmp91 = tmp90 * tmp0
    tmp92 = tmp91 + tmp0
    tmp93 = tmp92 * tmp0
    tmp94 = tmp93 + tmp0
    tmp95 = tmp94 * tmp0
    tmp96 = tmp95 + tmp0
    tmp97 = tmp96 * tmp0
    tmp98 = tmp97 + tmp0
    tmp99 = tmp98 * tmp0
    tmp100 = tmp99 + tmp0
    tmp101 = tmp100 * tmp0
    tmp102 = tmp101 + tmp0
    tmp103 = tmp102 * tmp0
    tmp104 = tmp103 + tmp0
    tmp105 = tmp104 * tmp0
    tmp106 = tmp105 + tmp0
    tmp107 = tmp106 * tmp0
    tmp108 = tmp107 + tmp0
    tmp109 = tmp108 * tmp0
    tmp110 = tmp109 + tmp0
    tmp111 = tmp110 * tmp0
    tmp112 = tmp111 + tmp0
    tmp113 = tmp112 * tmp0
    tmp114 = tmp113 + tmp0
    tmp115 = tmp114 * tmp0
    tmp116 = tmp115 + tmp0
    tmp117 = tmp116 * tmp0
    tmp118 = tmp117 + tmp0
    tmp119 = tmp118 * tmp0
    tmp120 = tmp119 + tmp0
    tmp121 = tmp120 * tmp0
    tmp122 = tmp121 + tmp0
    tmp123 = tmp122 * tmp0
    tmp124 = tmp123 + tmp0
    tmp125 = tmp124 * tmp0
    tmp126 = tmp125 + tmp0
    tmp127 = tmp126 * tmp0
    tmp128 = tmp127 + tmp0
    tl.store(in_out_ptr0 + (x0), tmp128, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, = args
        args.clear()
        assert_size_stride(arg0_1, (67108864, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((67108864, ), (1, ), torch.float32)
            buf1 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [mul, acc, mul_1, acc_1, mul_2, acc_2, mul_3, acc_3, mul_4, acc_4, mul_5, acc_5, mul_6, acc_6, mul_7, acc_7, mul_8, acc_8, mul_9, acc_9, mul_10, acc_10, mul_11, acc_11, mul_12, acc_12, mul_13, acc_13, mul_14, acc_14, mul_15, acc_15, mul_16, acc_16, mul_17, acc_17, mul_18, acc_18, mul_19, acc_19, mul_20, acc_20, mul_21, acc_21, mul_22, acc_22, mul_23, acc_23, mul_24, acc_24, mul_25, acc_25, mul_26, acc_26, mul_27, acc_27, mul_28, acc_28, mul_29, acc_29, mul_30, acc_30, mul_31, acc_31, mul_32, acc_32, mul_33, acc_33, mul_34, acc_34, mul_35, acc_35, mul_36, acc_36, mul_37, acc_37, mul_38, acc_38, mul_39, acc_39, mul_40, acc_40, mul_41, acc_41, mul_42, acc_42, mul_43, acc_43, mul_44, acc_44, mul_45, acc_45, mul_46, acc_46, mul_47, acc_47, mul_48, acc_48, mul_49, acc_49, mul_50, acc_50, mul_51, acc_51, mul_52, acc_52, mul_53, acc_53, mul_54, acc_54, mul_55, acc_55, mul_56, acc_56, mul_57, acc_57, mul_58, acc_58, mul_59, acc_59, mul_60, acc_60, mul_61, acc_61, mul_62, acc_62, mul_63, acc_63], Original ATen: [aten.mul, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_mul_0:7
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_0.run(buf1, arg0_1, 67108864, stream=stream0)
            del arg0_1
        return (buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((67108864, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
