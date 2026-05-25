# AOT ID: ['5_inference']
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


# kernel path: /tmp/torchinductor_gg/ak/cakbbbs6hkeffc6yzuqexedrzb34heezl37abgxwqqxrh3kz7cdq.py
# Topologically Sorted Source Nodes: [mul, acc, mul_1, acc_1, mul_2, acc_2, mul_3, acc_3, mul_4, acc_4, mul_5, acc_5, mul_6, acc_6, mul_7, acc_7, mul_8, acc_8, mul_9, acc_9, mul_10, acc_10, mul_11, acc_11, mul_12, acc_12, mul_13, acc_13, mul_14, acc_14, mul_15, acc_15, mul_16, acc_16, mul_17, acc_17, mul_18, acc_18, mul_19, acc_19, mul_20, acc_20, mul_21, acc_21, mul_22, acc_22, mul_23, acc_23, mul_24, acc_24, mul_25, acc_25, mul_26, acc_26, mul_27, acc_27, mul_28, acc_28, mul_29, acc_29, mul_30, acc_30, mul_31, acc_31], Original ATen: [aten.mul, aten.add]
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
#   acc_4 => add_4
#   acc_5 => add_5
#   acc_6 => add_6
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
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
# Graph fragment:
#   %arg0_1 : Tensor "f32[67108864][1]cuda:0" = PlaceHolder[target=arg0_1]
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
#   return %add_31
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=24, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '6E2CBEB9BFB65B06FFC57A5E69F68D54F759F46CC4C44428A4C003EBC7ADE616', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 805306368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x0), tmp64, None)
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
            # Topologically Sorted Source Nodes: [mul, acc, mul_1, acc_1, mul_2, acc_2, mul_3, acc_3, mul_4, acc_4, mul_5, acc_5, mul_6, acc_6, mul_7, acc_7, mul_8, acc_8, mul_9, acc_9, mul_10, acc_10, mul_11, acc_11, mul_12, acc_12, mul_13, acc_13, mul_14, acc_14, mul_15, acc_15, mul_16, acc_16, mul_17, acc_17, mul_18, acc_18, mul_19, acc_19, mul_20, acc_20, mul_21, acc_21, mul_22, acc_22, mul_23, acc_23, mul_24, acc_24, mul_25, acc_25, mul_26, acc_26, mul_27, acc_27, mul_28, acc_28, mul_29, acc_29, mul_30, acc_30, mul_31, acc_31], Original ATen: [aten.mul, aten.add]
            # [Provenance debug handles] triton_poi_fused_add_mul_0:6
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_0.run(arg0_1, buf0, 67108864, stream=stream0)
            del arg0_1
        return (buf0, )

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
