"""Compare repeated layer visits with GGUFs containing explicit copies of the block."""

import argparse
import ctypes as ct
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from zlib import crc32

import numpy as np


def make_model(path, physical):
    from gguf import GGUFValueType as T
    from gguf import GGUFWriter

    w = GGUFWriter(path, "deepseek41")
    w.add_tokenizer_model("none")
    scalars = {
        "vocab_size": 64,
        "block_count": len(physical),
        "context_length": 128,
        "embedding_length": 64,
        "attention.head_count": 2,
        "attention.head_count_kv": 1,
        "attention.key_length": 64,
        "attention.value_length": 64,
        "attention.q_lora_rank": 32,
        "attention.sliding_window": 4,
        "rope.dimension_count": 32,
        "expert_count": 4,
        "expert_used_count": 2,
        "expert_gating_func": 4,
        "expert_shared_count": 1,
        "expert_feed_forward_length": 64,
        "attention.indexer.head_count": 2,
        "attention.indexer.key_length": 64,
        "attention.indexer.top_k": 2,
        "attention.output_group_count": 2,
        "attention.output_lora_rank": 32,
        "hyper_connection.count": 4,
        "hyper_connection.sinkhorn_iterations": 10,
        "hash_layer_count": 0,
        "ple.ngram_size": 2,
        "ple.heads_per_ngram": 1,
        "embedding_length_per_layer_input": 16,
        "engram.pad_token_id": 0,
        "rope.freq_base": 10000.0,
        "attention.compress_rope_freq_base": 160000.0,
        "expert_weights_scale": 1.5,
        "expert_weights_norm": True,
        "attention.layer_norm_rms_epsilon": 1e-6,
        "hyper_connection.epsilon": 1e-6,
    }
    for k, v in scalars.items():
        kind = {int: T.UINT32, float: T.FLOAT32, bool: T.BOOL}[type(v)]
        w.add_key_value("deepseek41." + k, v, kind)
    # Layer 1 owns compressed rows, the index, and the Engram table lookup.
    # In the unrolled file its second copy has identical weights and hash salts.
    sources = [i for i, p in enumerate(physical) if p == 1]
    arrays = {
        "swiglu_clamp_exp": ([2.0 + p for p in physical], T.FLOAT32),
        "attention.compress_ratios": ([0 if p == 0 else 2 for p in physical], T.UINT32),
        "attention.kv_source_layers": (sources, T.UINT32),
        "attention.indexer.source_layers": (sources, T.UINT32),
        "ple.layers": (sources, T.UINT32),
        "ple.layer_multipliers": ([17, 31] * len(sources), T.UINT64),
        "ple.head_offsets": ([0] * len(sources), T.UINT64),
        "ple.head_vocab_sizes": ([67] * len(sources), T.UINT64),
        "engram.token_map": (list(range(64)), T.INT32),
    }
    for k, (v, t) in arrays.items():
        w.add_key_value("deepseek41." + k, v, T.ARRAY, sub_type=t)

    def tensor(name, shape, physical_name):
        rng = np.random.default_rng(crc32(physical_name.encode()))
        data = rng.standard_normal(shape[::-1], dtype=np.float32) / np.sqrt(
            np.float32(shape[0])
        )
        if "norm.weight" in name:
            data = np.ones_like(data)
        w.add_tensor(name, data)

    for name, shape in {
        "token_embd": (64, 64),
        "output_norm": (64,),
        "output": (64, 64),
        "per_layer_token_embd": (16, 67),
    }.items():
        tensor(name + ".weight", shape, name)
    for il, p in enumerate(physical):
        shapes = {
            "attn_norm": (64,),
            "attn_sinks": (2,),
            "attn_q_a": (64, 32),
            "attn_q_a_norm": (32,),
            "attn_q_b": (32, 128),
            "attn_kv": (64, 64),
            "attn_kv_a_norm": (64,),
            "attn_output_a": (64, 32, 2),
            "attn_output_b": (64, 64),
            "ffn_gate_inp": (64, 4),
            "exp_probs_b": (4,),
            "ffn_norm": (64,),
        }
        for sub in ["attn", "ffn"]:
            shapes.update(
                {
                    f"hc_{sub}_fn": (256, 24),
                    f"hc_{sub}_base": (24,),
                    f"hc_{sub}_scale": (3,),
                }
            )
        for proj in ["up", "gate", "down"]:
            shapes[f"ffn_{proj}_exps"] = (64, 64, 4)
            shapes[f"ffn_{proj}_shexp"] = (64, 64)
        if p == 1:
            shapes.update(
                {
                    "attn_compressor_kv": (64, 64),
                    "attn_compressor_norm": (64,),
                    "attn_compressor_gate": (64, 64),
                    "indexer.attn_k": (64, 64),
                    "indexer.k_norm": (64,),
                    "indexer.attn_q_b": (32, 128),
                    "indexer.proj": (64, 2),
                    "engram_kv": (16, 320),
                    "engram_gate": (64, 4),
                }
            )
        for name, shape in shapes.items():
            suffix = "bias" if name == "exp_probs_b" else "weight"
            tensor(f"blk.{il}.{name}.{suffix}", shape, f"{p}.{name}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def evaluate(lib, path, window, gpu, repeats=None):
    if repeats is None:
        os.environ.pop("DS4_N1X_LOOP_REPEATS", None)
    else:
        os.environ["DS4_N1X_LOOP_REPEATS"] = str(repeats)
    if window is None:
        os.environ.pop("DS4_N1X_LOOP_LAYERS", None)
    else:
        os.environ["DS4_N1X_LOOP_LAYERS"] = window
    handle = lib.ds4_bridge_gguf_open(os.fsencode(path), int(gpu), 128, 2, 64)
    if not handle:
        raise RuntimeError(f"could not load {path}")
    try:
        logits = []
        for ids in ([1, 4, 2, 6, 3, 9, 5, 8, 3], [6], [3], [11], [20, 15, 30]):
            tokens = (ct.c_int * len(ids))(*ids)
            out = lib.ds4_bridge_gguf_eval(handle, tokens, len(ids))
            if not out:
                raise RuntimeError("evaluation failed")
            logits.append(np.ctypeslib.as_array(out, shape=(64,)).copy())
        return np.stack(logits)
    finally:
        lib.ds4_bridge_gguf_close(handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("target/n1x"))
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    build = args.build.resolve()
    sys.path.insert(0, str(build / "llama-src/gguf-py"))
    with ExitStack() as stack:
        stack.enter_context(os.add_dll_directory(str(build)))
        stack.enter_context(
            os.add_dll_directory(str(Path(os.environ["CUDA_PATH"]) / "bin/arm64"))
        )
        tmp = stack.enter_context(tempfile.TemporaryDirectory())
        lib = ct.CDLL(str(build / "ds41_gguf.dll"))
        lib.ds4_bridge_gguf_open.argtypes = [ct.c_char_p] + [ct.c_int] * 4
        lib.ds4_bridge_gguf_open.restype = ct.c_void_p
        lib.ds4_bridge_gguf_eval.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_int),
            ct.c_int,
        ]
        lib.ds4_bridge_gguf_eval.restype = ct.POINTER(ct.c_float)
        lib.ds4_bridge_gguf_close.argtypes = [ct.c_void_p]
        lib.ds4_bridge_gguf_close.restype = None
        small = Path(tmp) / "four.gguf"
        make_model(small, [0, 1, 2, 3])
        ordinary = evaluate(lib, small, None, args.cuda)
        zero = evaluate(lib, small, "1-2", args.cuda, 0)
        np.testing.assert_allclose(zero, ordinary, atol=1e-5, rtol=1e-5)
        for repeats in (1, 2, 3):
            unrolled = Path(tmp) / f"unrolled-{repeats}.gguf"
            make_model(unrolled, [0] + [1, 2] * (1 + repeats) + [3])
            expected = evaluate(lib, unrolled, None, args.cuda)
            # An unset repeat count preserves the original one-repeat configuration.
            setting = None if repeats == 1 else repeats
            with patch.dict(os.environ, {"DS4_N1X_SHARE_LOOP_EXPERTS": "1"}):
                actual = evaluate(lib, small, "1-2", args.cuda, setting)
            if args.cuda:
                with patch.dict(os.environ, {"DS4_N1X_SHARE_LOOP_EXPERTS": "0"}):
                    separate = evaluate(lib, small, "1-2", True, setting)
                np.testing.assert_allclose(actual, separate, atol=1e-5, rtol=1e-5)
            assert np.isfinite([expected, actual, ordinary]).all()
            assert not np.allclose(expected, ordinary)
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            print(f"{repeats} extra passes, 64 logits at five frontiers, "
                  f"maximum absolute difference {np.max(np.abs(actual - expected))}")
