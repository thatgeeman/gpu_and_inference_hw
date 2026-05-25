# AOT ID: ['7_inference']
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


# kernel path: /tmp/torchinductor_gg/ka/ckaqqbuwi64wzcsuwnrjvnkuvfdf6htdzfkfkft4kznjpsyrol5x.py
# Topologically Sorted Source Nodes: [mul, acc, mul_1, acc_1, mul_2, acc_2, mul_3, acc_3, mul_4, acc_4, mul_5, acc_5, mul_6, acc_6, mul_7, acc_7, mul_8, acc_8, mul_9, acc_9, mul_10, acc_10, mul_11, acc_11, mul_12, acc_12, mul_13, acc_13, mul_14, acc_14, mul_15, acc_15, mul_16, acc_16, mul_17, acc_17, mul_18, acc_18, mul_19, acc_19, mul_20, acc_20, mul_21, acc_21, mul_22, acc_22, mul_23, acc_23, mul_24, acc_24, mul_25, acc_25, mul_26, acc_26, mul_27, acc_27, mul_28, acc_28, mul_29, acc_29, mul_30, acc_30, mul_31, acc_31, mul_32, acc_32, mul_33, acc_33, mul_34, acc_34, mul_35, acc_35, mul_36, acc_36, mul_37, acc_37, mul_38, acc_38, mul_39, acc_39, mul_40, acc_40, mul_41, acc_41, mul_42, acc_42, mul_43, acc_43, mul_44, acc_44, mul_45, acc_45, mul_46, acc_46, mul_47, acc_47, mul_48, acc_48, mul_49, acc_49, mul_50, acc_50, mul_51, acc_51, mul_52, acc_52, mul_53, acc_53, mul_54, acc_54, mul_55, acc_55, mul_56, acc_56, mul_57, acc_57, mul_58, acc_58, mul_59, acc_59, mul_60, acc_60, mul_61, acc_61, mul_62, acc_62, mul_63, acc_63, mul_64, acc_64, mul_65, acc_65, mul_66, acc_66, mul_67, acc_67, mul_68, acc_68, mul_69, acc_69, mul_70, acc_70, mul_71, acc_71, mul_72, acc_72, mul_73, acc_73, mul_74, acc_74, mul_75, acc_75, mul_76, acc_76, mul_77, acc_77, mul_78, acc_78, mul_79, acc_79, mul_80, acc_80, mul_81, acc_81, mul_82, acc_82, mul_83, acc_83, mul_84, acc_84, mul_85, acc_85, mul_86, acc_86, mul_87, acc_87, mul_88, acc_88, mul_89, acc_89, mul_90, acc_90, mul_91, acc_91, mul_92, acc_92, mul_93, acc_93, mul_94, acc_94, mul_95, acc_95, mul_96, acc_96, mul_97, acc_97, mul_98, acc_98, mul_99, acc_99, mul_100, acc_100, mul_101, acc_101, mul_102, acc_102, mul_103, acc_103, mul_104, acc_104, mul_105, acc_105, mul_106, acc_106, mul_107, acc_107, mul_108, acc_108, mul_109, acc_109, mul_110, acc_110, mul_111, acc_111, mul_112, acc_112, mul_113, acc_113, mul_114, acc_114, mul_115, acc_115, mul_116, acc_116, mul_117, acc_117, mul_118, acc_118, mul_119, acc_119, mul_120, acc_120, mul_121, acc_121, mul_122, acc_122, mul_123, acc_123, mul_124, acc_124, mul_125, acc_125, mul_126, acc_126, mul_127, acc_127], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   acc => add
#   acc_1 => add_1
#   acc_10 => add_10
#   acc_100 => add_100
#   acc_101 => add_101
#   acc_102 => add_102
#   acc_103 => add_103
#   acc_104 => add_104
#   acc_105 => add_105
#   acc_106 => add_106
#   acc_107 => add_107
#   acc_108 => add_108
#   acc_109 => add_109
#   acc_11 => add_11
#   acc_110 => add_110
#   acc_111 => add_111
#   acc_112 => add_112
#   acc_113 => add_113
#   acc_114 => add_114
#   acc_115 => add_115
#   acc_116 => add_116
#   acc_117 => add_117
#   acc_118 => add_118
#   acc_119 => add_119
#   acc_12 => add_12
#   acc_120 => add_120
#   acc_121 => add_121
#   acc_122 => add_122
#   acc_123 => add_123
#   acc_124 => add_124
#   acc_125 => add_125
#   acc_126 => add_126
#   acc_127 => add_127
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
#   acc_64 => add_64
#   acc_65 => add_65
#   acc_66 => add_66
#   acc_67 => add_67
#   acc_68 => add_68
#   acc_69 => add_69
#   acc_7 => add_7
#   acc_70 => add_70
#   acc_71 => add_71
#   acc_72 => add_72
#   acc_73 => add_73
#   acc_74 => add_74
#   acc_75 => add_75
#   acc_76 => add_76
#   acc_77 => add_77
#   acc_78 => add_78
#   acc_79 => add_79
#   acc_8 => add_8
#   acc_80 => add_80
#   acc_81 => add_81
#   acc_82 => add_82
#   acc_83 => add_83
#   acc_84 => add_84
#   acc_85 => add_85
#   acc_86 => add_86
#   acc_87 => add_87
#   acc_88 => add_88
#   acc_89 => add_89
#   acc_9 => add_9
#   acc_90 => add_90
#   acc_91 => add_91
#   acc_92 => add_92
#   acc_93 => add_93
#   acc_94 => add_94
#   acc_95 => add_95
#   acc_96 => add_96
#   acc_97 => add_97
#   acc_98 => add_98
#   acc_99 => add_99
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_100 => mul_100
#   mul_101 => mul_101
#   mul_102 => mul_102
#   mul_103 => mul_103
#   mul_104 => mul_104
#   mul_105 => mul_105
#   mul_106 => mul_106
#   mul_107 => mul_107
#   mul_108 => mul_108
#   mul_109 => mul_109
#   mul_11 => mul_11
#   mul_110 => mul_110
#   mul_111 => mul_111
#   mul_112 => mul_112
#   mul_113 => mul_113
#   mul_114 => mul_114
#   mul_115 => mul_115
#   mul_116 => mul_116
#   mul_117 => mul_117
#   mul_118 => mul_118
#   mul_119 => mul_119
#   mul_12 => mul_12
#   mul_120 => mul_120
#   mul_121 => mul_121
#   mul_122 => mul_122
#   mul_123 => mul_123
#   mul_124 => mul_124
#   mul_125 => mul_125
#   mul_126 => mul_126
#   mul_127 => mul_127
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
#   mul_64 => mul_64
#   mul_65 => mul_65
#   mul_66 => mul_66
#   mul_67 => mul_67
#   mul_68 => mul_68
#   mul_69 => mul_69
#   mul_7 => mul_7
#   mul_70 => mul_70
#   mul_71 => mul_71
#   mul_72 => mul_72
#   mul_73 => mul_73
#   mul_74 => mul_74
#   mul_75 => mul_75
#   mul_76 => mul_76
#   mul_77 => mul_77
#   mul_78 => mul_78
#   mul_79 => mul_79
#   mul_8 => mul_8
#   mul_80 => mul_80
#   mul_81 => mul_81
#   mul_82 => mul_82
#   mul_83 => mul_83
#   mul_84 => mul_84
#   mul_85 => mul_85
#   mul_86 => mul_86
#   mul_87 => mul_87
#   mul_88 => mul_88
#   mul_89 => mul_89
#   mul_9 => mul_9
#   mul_90 => mul_90
#   mul_91 => mul_91
#   mul_92 => mul_92
#   mul_93 => mul_93
#   mul_94 => mul_94
#   mul_95 => mul_95
#   mul_96 => mul_96
#   mul_97 => mul_97
#   mul_98 => mul_98
#   mul_99 => mul_99
# Graph fragment:
#   %arg0_1 : Tensor "f32[67108864][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %add_49 : Tensor "f32[67108864][1]cuda:0" = PlaceHolder[target=add_49]
#   %mul_64 : Tensor "f32[67108864][1]cuda:0" = PlaceHolder[target=mul_64]
#   %add_78 : Tensor "f32[67108864][1]cuda:0" = PlaceHolder[target=add_78]
#   %mul_93 : Tensor "f32[67108864][1]cuda:0" = PlaceHolder[target=mul_93]
#   %add_107 : Tensor "f32[67108864][1]cuda:0" = PlaceHolder[target=add_107]
#   %mul_122 : Tensor "f32[67108864][1]cuda:0" = PlaceHolder[target=mul_122]
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
#   %mul_64 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_63, %arg0_1), kwargs = {})
#   %add_64 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_64, %arg0_1), kwargs = {})
#   %mul_65 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_64, %arg0_1), kwargs = {})
#   %add_65 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %arg0_1), kwargs = {})
#   %mul_66 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_65, %arg0_1), kwargs = {})
#   %add_66 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_66, %arg0_1), kwargs = {})
#   %mul_67 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_66, %arg0_1), kwargs = {})
#   %add_67 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_67, %arg0_1), kwargs = {})
#   %mul_68 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_67, %arg0_1), kwargs = {})
#   %add_68 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %arg0_1), kwargs = {})
#   %mul_69 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_68, %arg0_1), kwargs = {})
#   %add_69 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_69, %arg0_1), kwargs = {})
#   %mul_70 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_69, %arg0_1), kwargs = {})
#   %add_70 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %arg0_1), kwargs = {})
#   %mul_71 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_70, %arg0_1), kwargs = {})
#   %add_71 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %arg0_1), kwargs = {})
#   %mul_72 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_71, %arg0_1), kwargs = {})
#   %add_72 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_72, %arg0_1), kwargs = {})
#   %mul_73 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_72, %arg0_1), kwargs = {})
#   %add_73 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_73, %arg0_1), kwargs = {})
#   %mul_74 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_73, %arg0_1), kwargs = {})
#   %add_74 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %arg0_1), kwargs = {})
#   %mul_75 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %arg0_1), kwargs = {})
#   %add_75 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, %arg0_1), kwargs = {})
#   %mul_76 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_75, %arg0_1), kwargs = {})
#   %add_76 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_76, %arg0_1), kwargs = {})
#   %mul_77 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_76, %arg0_1), kwargs = {})
#   %add_77 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %arg0_1), kwargs = {})
#   %mul_78 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_77, %arg0_1), kwargs = {})
#   %add_78 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_78, %arg0_1), kwargs = {})
#   %mul_79 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_78, %arg0_1), kwargs = {})
#   %add_79 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_79, %arg0_1), kwargs = {})
#   %mul_80 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %arg0_1), kwargs = {})
#   %add_80 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %arg0_1), kwargs = {})
#   %mul_81 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_80, %arg0_1), kwargs = {})
#   %add_81 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_81, %arg0_1), kwargs = {})
#   %mul_82 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_81, %arg0_1), kwargs = {})
#   %add_82 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_82, %arg0_1), kwargs = {})
#   %mul_83 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_82, %arg0_1), kwargs = {})
#   %add_83 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %arg0_1), kwargs = {})
#   %mul_84 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_83, %arg0_1), kwargs = {})
#   %add_84 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, %arg0_1), kwargs = {})
#   %mul_85 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_84, %arg0_1), kwargs = {})
#   %add_85 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_85, %arg0_1), kwargs = {})
#   %mul_86 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_85, %arg0_1), kwargs = {})
#   %add_86 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %arg0_1), kwargs = {})
#   %mul_87 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_86, %arg0_1), kwargs = {})
#   %add_87 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %arg0_1), kwargs = {})
#   %mul_88 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_87, %arg0_1), kwargs = {})
#   %add_88 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_88, %arg0_1), kwargs = {})
#   %mul_89 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_88, %arg0_1), kwargs = {})
#   %add_89 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %arg0_1), kwargs = {})
#   %mul_90 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_89, %arg0_1), kwargs = {})
#   %add_90 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %arg0_1), kwargs = {})
#   %mul_91 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_90, %arg0_1), kwargs = {})
#   %add_91 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %arg0_1), kwargs = {})
#   %mul_92 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_91, %arg0_1), kwargs = {})
#   %add_92 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_92, %arg0_1), kwargs = {})
#   %mul_93 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_92, %arg0_1), kwargs = {})
#   %add_93 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_93, %arg0_1), kwargs = {})
#   %mul_94 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_93, %arg0_1), kwargs = {})
#   %add_94 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %arg0_1), kwargs = {})
#   %mul_95 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_94, %arg0_1), kwargs = {})
#   %add_95 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_95, %arg0_1), kwargs = {})
#   %mul_96 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_95, %arg0_1), kwargs = {})
#   %add_96 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_96, %arg0_1), kwargs = {})
#   %mul_97 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_96, %arg0_1), kwargs = {})
#   %add_97 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %arg0_1), kwargs = {})
#   %mul_98 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_97, %arg0_1), kwargs = {})
#   %add_98 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_98, %arg0_1), kwargs = {})
#   %mul_99 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_98, %arg0_1), kwargs = {})
#   %add_99 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_99, %arg0_1), kwargs = {})
#   %mul_100 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_99, %arg0_1), kwargs = {})
#   %add_100 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_100, %arg0_1), kwargs = {})
#   %mul_101 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_100, %arg0_1), kwargs = {})
#   %add_101 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %arg0_1), kwargs = {})
#   %mul_102 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_101, %arg0_1), kwargs = {})
#   %add_102 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_102, %arg0_1), kwargs = {})
#   %mul_103 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_102, %arg0_1), kwargs = {})
#   %add_103 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_103, %arg0_1), kwargs = {})
#   %mul_104 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_103, %arg0_1), kwargs = {})
#   %add_104 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_104, %arg0_1), kwargs = {})
#   %mul_105 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_104, %arg0_1), kwargs = {})
#   %add_105 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_105, %arg0_1), kwargs = {})
#   %mul_106 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_105, %arg0_1), kwargs = {})
#   %add_106 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_106, %arg0_1), kwargs = {})
#   %mul_107 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_106, %arg0_1), kwargs = {})
#   %add_107 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_107, %arg0_1), kwargs = {})
#   %mul_108 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_107, %arg0_1), kwargs = {})
#   %add_108 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_108, %arg0_1), kwargs = {})
#   %mul_109 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_108, %arg0_1), kwargs = {})
#   %add_109 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_109, %arg0_1), kwargs = {})
#   %mul_110 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_109, %arg0_1), kwargs = {})
#   %add_110 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %arg0_1), kwargs = {})
#   %mul_111 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_110, %arg0_1), kwargs = {})
#   %add_111 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_111, %arg0_1), kwargs = {})
#   %mul_112 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_111, %arg0_1), kwargs = {})
#   %add_112 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_112, %arg0_1), kwargs = {})
#   %mul_113 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_112, %arg0_1), kwargs = {})
#   %add_113 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %arg0_1), kwargs = {})
#   %mul_114 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_113, %arg0_1), kwargs = {})
#   %add_114 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_114, %arg0_1), kwargs = {})
#   %mul_115 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_114, %arg0_1), kwargs = {})
#   %add_115 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_115, %arg0_1), kwargs = {})
#   %mul_116 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_115, %arg0_1), kwargs = {})
#   %add_116 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, %arg0_1), kwargs = {})
#   %mul_117 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_116, %arg0_1), kwargs = {})
#   %add_117 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_117, %arg0_1), kwargs = {})
#   %mul_118 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_117, %arg0_1), kwargs = {})
#   %add_118 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_118, %arg0_1), kwargs = {})
#   %mul_119 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_118, %arg0_1), kwargs = {})
#   %add_119 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %arg0_1), kwargs = {})
#   %mul_120 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_119, %arg0_1), kwargs = {})
#   %add_120 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_120, %arg0_1), kwargs = {})
#   %mul_121 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_120, %arg0_1), kwargs = {})
#   %add_121 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_121, %arg0_1), kwargs = {})
#   %mul_122 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_121, %arg0_1), kwargs = {})
#   %add_122 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_122, %arg0_1), kwargs = {})
#   %mul_123 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_122, %arg0_1), kwargs = {})
#   %add_123 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_123, %arg0_1), kwargs = {})
#   %mul_124 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_123, %arg0_1), kwargs = {})
#   %add_124 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_124, %arg0_1), kwargs = {})
#   %mul_125 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_124, %arg0_1), kwargs = {})
#   %add_125 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %arg0_1), kwargs = {})
#   %mul_126 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_125, %arg0_1), kwargs = {})
#   %add_126 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_126, %arg0_1), kwargs = {})
#   %mul_127 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_126, %arg0_1), kwargs = {})
#   %add_127 : Tensor "f32[67108864][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_127, %arg0_1), kwargs = {})
#   return %add_49,%mul_64,%add_78,%mul_93,%add_107,%mul_122,%add_127
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
    tmp129 = tmp128 * tmp0
    tmp130 = tmp129 + tmp0
    tmp131 = tmp130 * tmp0
    tmp132 = tmp131 + tmp0
    tmp133 = tmp132 * tmp0
    tmp134 = tmp133 + tmp0
    tmp135 = tmp134 * tmp0
    tmp136 = tmp135 + tmp0
    tmp137 = tmp136 * tmp0
    tmp138 = tmp137 + tmp0
    tmp139 = tmp138 * tmp0
    tmp140 = tmp139 + tmp0
    tmp141 = tmp140 * tmp0
    tmp142 = tmp141 + tmp0
    tmp143 = tmp142 * tmp0
    tmp144 = tmp143 + tmp0
    tmp145 = tmp144 * tmp0
    tmp146 = tmp145 + tmp0
    tmp147 = tmp146 * tmp0
    tmp148 = tmp147 + tmp0
    tmp149 = tmp148 * tmp0
    tmp150 = tmp149 + tmp0
    tmp151 = tmp150 * tmp0
    tmp152 = tmp151 + tmp0
    tmp153 = tmp152 * tmp0
    tmp154 = tmp153 + tmp0
    tmp155 = tmp154 * tmp0
    tmp156 = tmp155 + tmp0
    tmp157 = tmp156 * tmp0
    tmp158 = tmp157 + tmp0
    tmp159 = tmp158 * tmp0
    tmp160 = tmp159 + tmp0
    tmp161 = tmp160 * tmp0
    tmp162 = tmp161 + tmp0
    tmp163 = tmp162 * tmp0
    tmp164 = tmp163 + tmp0
    tmp165 = tmp164 * tmp0
    tmp166 = tmp165 + tmp0
    tmp167 = tmp166 * tmp0
    tmp168 = tmp167 + tmp0
    tmp169 = tmp168 * tmp0
    tmp170 = tmp169 + tmp0
    tmp171 = tmp170 * tmp0
    tmp172 = tmp171 + tmp0
    tmp173 = tmp172 * tmp0
    tmp174 = tmp173 + tmp0
    tmp175 = tmp174 * tmp0
    tmp176 = tmp175 + tmp0
    tmp177 = tmp176 * tmp0
    tmp178 = tmp177 + tmp0
    tmp179 = tmp178 * tmp0
    tmp180 = tmp179 + tmp0
    tmp181 = tmp180 * tmp0
    tmp182 = tmp181 + tmp0
    tmp183 = tmp182 * tmp0
    tmp184 = tmp183 + tmp0
    tmp185 = tmp184 * tmp0
    tmp186 = tmp185 + tmp0
    tmp187 = tmp186 * tmp0
    tmp188 = tmp187 + tmp0
    tmp189 = tmp188 * tmp0
    tmp190 = tmp189 + tmp0
    tmp191 = tmp190 * tmp0
    tmp192 = tmp191 + tmp0
    tmp193 = tmp192 * tmp0
    tmp194 = tmp193 + tmp0
    tmp195 = tmp194 * tmp0
    tmp196 = tmp195 + tmp0
    tmp197 = tmp196 * tmp0
    tmp198 = tmp197 + tmp0
    tmp199 = tmp198 * tmp0
    tmp200 = tmp199 + tmp0
    tmp201 = tmp200 * tmp0
    tmp202 = tmp201 + tmp0
    tmp203 = tmp202 * tmp0
    tmp204 = tmp203 + tmp0
    tmp205 = tmp204 * tmp0
    tmp206 = tmp205 + tmp0
    tmp207 = tmp206 * tmp0
    tmp208 = tmp207 + tmp0
    tmp209 = tmp208 * tmp0
    tmp210 = tmp209 + tmp0
    tmp211 = tmp210 * tmp0
    tmp212 = tmp211 + tmp0
    tmp213 = tmp212 * tmp0
    tmp214 = tmp213 + tmp0
    tmp215 = tmp214 * tmp0
    tmp216 = tmp215 + tmp0
    tmp217 = tmp216 * tmp0
    tmp218 = tmp217 + tmp0
    tmp219 = tmp218 * tmp0
    tmp220 = tmp219 + tmp0
    tmp221 = tmp220 * tmp0
    tmp222 = tmp221 + tmp0
    tmp223 = tmp222 * tmp0
    tmp224 = tmp223 + tmp0
    tmp225 = tmp224 * tmp0
    tmp226 = tmp225 + tmp0
    tmp227 = tmp226 * tmp0
    tmp228 = tmp227 + tmp0
    tmp229 = tmp228 * tmp0
    tmp230 = tmp229 + tmp0
    tmp231 = tmp230 * tmp0
    tmp232 = tmp231 + tmp0
    tmp233 = tmp232 * tmp0
    tmp234 = tmp233 + tmp0
    tmp235 = tmp234 * tmp0
    tmp236 = tmp235 + tmp0
    tmp237 = tmp236 * tmp0
    tmp238 = tmp237 + tmp0
    tmp239 = tmp238 * tmp0
    tmp240 = tmp239 + tmp0
    tmp241 = tmp240 * tmp0
    tmp242 = tmp241 + tmp0
    tmp243 = tmp242 * tmp0
    tmp244 = tmp243 + tmp0
    tmp245 = tmp244 * tmp0
    tmp246 = tmp245 + tmp0
    tmp247 = tmp246 * tmp0
    tmp248 = tmp247 + tmp0
    tmp249 = tmp248 * tmp0
    tmp250 = tmp249 + tmp0
    tmp251 = tmp250 * tmp0
    tmp252 = tmp251 + tmp0
    tmp253 = tmp252 * tmp0
    tmp254 = tmp253 + tmp0
    tmp255 = tmp254 * tmp0
    tmp256 = tmp255 + tmp0
    tl.store(in_out_ptr0 + (x0), tmp256, None)
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
            buf2 = buf1; del buf1  # reuse
            buf3 = buf2; del buf2  # reuse
            buf4 = buf3; del buf3  # reuse
            buf5 = buf4; del buf4  # reuse
            buf6 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [mul, acc, mul_1, acc_1, mul_2, acc_2, mul_3, acc_3, mul_4, acc_4, mul_5, acc_5, mul_6, acc_6, mul_7, acc_7, mul_8, acc_8, mul_9, acc_9, mul_10, acc_10, mul_11, acc_11, mul_12, acc_12, mul_13, acc_13, mul_14, acc_14, mul_15, acc_15, mul_16, acc_16, mul_17, acc_17, mul_18, acc_18, mul_19, acc_19, mul_20, acc_20, mul_21, acc_21, mul_22, acc_22, mul_23, acc_23, mul_24, acc_24, mul_25, acc_25, mul_26, acc_26, mul_27, acc_27, mul_28, acc_28, mul_29, acc_29, mul_30, acc_30, mul_31, acc_31, mul_32, acc_32, mul_33, acc_33, mul_34, acc_34, mul_35, acc_35, mul_36, acc_36, mul_37, acc_37, mul_38, acc_38, mul_39, acc_39, mul_40, acc_40, mul_41, acc_41, mul_42, acc_42, mul_43, acc_43, mul_44, acc_44, mul_45, acc_45, mul_46, acc_46, mul_47, acc_47, mul_48, acc_48, mul_49, acc_49, mul_50, acc_50, mul_51, acc_51, mul_52, acc_52, mul_53, acc_53, mul_54, acc_54, mul_55, acc_55, mul_56, acc_56, mul_57, acc_57, mul_58, acc_58, mul_59, acc_59, mul_60, acc_60, mul_61, acc_61, mul_62, acc_62, mul_63, acc_63, mul_64, acc_64, mul_65, acc_65, mul_66, acc_66, mul_67, acc_67, mul_68, acc_68, mul_69, acc_69, mul_70, acc_70, mul_71, acc_71, mul_72, acc_72, mul_73, acc_73, mul_74, acc_74, mul_75, acc_75, mul_76, acc_76, mul_77, acc_77, mul_78, acc_78, mul_79, acc_79, mul_80, acc_80, mul_81, acc_81, mul_82, acc_82, mul_83, acc_83, mul_84, acc_84, mul_85, acc_85, mul_86, acc_86, mul_87, acc_87, mul_88, acc_88, mul_89, acc_89, mul_90, acc_90, mul_91, acc_91, mul_92, acc_92, mul_93, acc_93, mul_94, acc_94, mul_95, acc_95, mul_96, acc_96, mul_97, acc_97, mul_98, acc_98, mul_99, acc_99, mul_100, acc_100, mul_101, acc_101, mul_102, acc_102, mul_103, acc_103, mul_104, acc_104, mul_105, acc_105, mul_106, acc_106, mul_107, acc_107, mul_108, acc_108, mul_109, acc_109, mul_110, acc_110, mul_111, acc_111, mul_112, acc_112, mul_113, acc_113, mul_114, acc_114, mul_115, acc_115, mul_116, acc_116, mul_117, acc_117, mul_118, acc_118, mul_119, acc_119, mul_120, acc_120, mul_121, acc_121, mul_122, acc_122, mul_123, acc_123, mul_124, acc_124, mul_125, acc_125, mul_126, acc_126, mul_127, acc_127], Original ATen: [aten.mul, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_mul_0:8
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_0.run(buf6, arg0_1, 67108864, stream=stream0)
            del arg0_1
        return (buf6, )

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
