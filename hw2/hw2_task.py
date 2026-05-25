from functools import partial

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from utils import (
    MAX_NEW_TOKENS,
    MODEL_NAME,
    PROFILE_STEPS,
    PROMPT_LEN,
    RESULTS_DIR,
    VOCAB_SIZE,
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
)


def optimized_loop(model, input_ids, n_steps):
    # TODO: fix the performance issues you found — changes may include
    # both `optimized_loop` and `generate_optimized`
    generated_tokens = []

    # First pass to model with full prompt to seed the KV cache
    outputs = model(input_ids=input_ids, use_cache=True)
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    generated_tokens.append(next_token_id.item())
    past_key_values = outputs.past_key_values
    print("Generated token 1:", next_token_id.item())
    # pass only the new token, so shape stays [1, 1] every step
    for _ in range(n_steps - 1):
        outputs = model(
            input_ids=next_token_id.unsqueeze(1),  # [batch, 1]
            # keys_to_ignore_at_inference is handled internally by the model when
            # using the past_key_values, so we don't need to manage it ourselves here
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token_id.item())
        past_key_values = outputs.past_key_values

    return generated_tokens


def trace_handler(prof, trace_name: str = "trace.json"):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))


def profile(loop_fn, model, input_ids, trace_name: str):
    # TODO: wrap loop_fn(model, input_ids, PROFILE_STEPS) with torch.profiler,
    # print the summary table, and export a Chrome trace to RESULTS_DIR / trace_name
    # based on https://docs.pytorch.org/docs/2.11/profiler.html#torch.profiler.profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,  # only for compiled, doc says not working for eager mode
        on_trace_ready=partial(trace_handler, trace_name=trace_name),
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)


def generate_optimized(optimized_trace_name: str) -> float:
    # TODO: load the model (consider dtype and other loading options),
    # then call profile() and time_generation() on optimized_loop.
    # Return the elapsed time from time_generation so main() can print a speedup.
    model = build_kv_cache_model(
        torch.float16
    )  # change dtype to float16 for faster inference
    input_ids = get_input_ids()
    # do a warmup run to exclude one-time setup costs from the profile and timing
    # optimized_loop(model, input_ids, n_steps=1)
    # profile the optimized loop to generate the trace
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    optimized_elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    del model
    torch.cuda.empty_cache()
    return optimized_elapsed


def build_kv_cache_model(dtype):
    """Model loader with optimizations for faster autoregressive generation,
    such as enabling the KV cache and using a faster dtype."""
    config = LlamaConfig(
        use_cache=True,  # enable KV cache
        vocab_size=VOCAB_SIZE,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=PROMPT_LEN + MAX_NEW_TOKENS + 64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config)
    model.to(device="cuda", dtype=dtype)
    model.eval()
    return model


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(
        optimized_trace_name="v1_optimized_trace.json"
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print(
            "generate_optimized() did not return a positive elapsed time; "
            "cannot compute speedup."
        )
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
#
# Biggest impact and why:
#
