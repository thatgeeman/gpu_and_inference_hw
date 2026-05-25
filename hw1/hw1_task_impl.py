import os

import numpy as np
import torch

os.environ["TORCH_COMPILE_DEBUG"] = "1"  # Enable Inductor debug logging for this task
os.environ["TORCH_COMPILE_LOG_LEVEL"] = (
    "debug"  # Set log level to debug for detailed output
)
os.environ["TORCHDYNAMO_VERBOSE"] = (
    "1"  # Enable verbose logging for TorchDynamo to see compilation details
)
# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    # TODO (1 line): implement a lowest-AI op
    # You do zero floating point operations but still move bytes across the memory bus.
    x = x.clone()
    return x


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    # TODO (1 line): return either `fn` or `torch.compile(fn)` based on `compiled`
    if compiled:
        return torch.compile(fn)
    else:
        return fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # TODO: time `rep` runs using CUDA events and return median latency (ms)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in range(rep):
        start_event.record()
        fn(*args)
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    return float(np.median(latencies))


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # TODO: compute total FLOPs, arithmetic intensity, and achieved FLOP/s
    if variant == "compiled":
        # Each loop iteration does 2 FLOPs per element (1 multiply + 1 add)
        flops_per_element = 2 * num_ops
        total_flops = flops_per_element * num_elements

        # Each element is read once and written once at the kernel boundary
        bytes_moved = 2 * bytes_per_element * num_elements  # read + write
    elif variant == "eager":
        # Each loop iteration launches separate multiply and add ops, so we have:
        # - num_ops multiplies: each reads acc and x, writes an intermediate tensor
        # - num_ops adds: each reads the intermediate tensor and x, writes to acc
        flops_per_element = 2 * num_ops
        total_flops = flops_per_element * num_elements

        # For memory traffic, we have to account for all the intermediate tensors:
        # - For multiplies: read acc and x, write intermediate (3 accesses per op)
        # - For adds: read intermediate and x, write acc (3 accesses per op)
        bytes_moved = (3 * num_ops) * bytes_per_element * num_elements
    else:
        raise ValueError(f"Unknown variant: {variant}")

    ai = total_flops / bytes_moved
    achieved_flops = total_flops / (ms * 1e-3)

    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# For this simple, static loop, `torch.compile` can capture and fuse the work
# into a much smaller number of kernels, keeping the bytes moved constant, regardless of op count.
# The runtime changes only little as the kernel is memory bound throughout, ie time needed is
# proportional to only the bytes moved, which is constant across the `1 ops` to `64 ops` points.
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# A 1024×1024 matmul produces a relatively small output tile workload.
# On an H100 with 132 streaming multiprocessors (SM), the tile grid may not fill all SMs evenly, leading to low utilization.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# This indicates that the computation is transitioning from being memory-bound to compute-bound.
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
# Eager ops launch separate kernels for each operation, leading to much higher memory traffic
# due to intermediate tensors, which significantly reduces performance compared to the fused
# compiled version.
