"""Compare native Varloop logits against the pinned llama.cpp graph on generated GGUFs."""

import argparse
import subprocess
import sys
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from zlib import crc32

import numpy as np


def make_model(path, passes, quantized, *, expand=False, tokenizer=False):
    from gguf import GGMLQuantizationType as Q
    from gguf import GGUFValueType as T
    from gguf import GGUFWriter, quantize

    w = GGUFWriter(path, "inkling2_loop")
    if tokenizer:
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        special = [
            "<|begin_of_text|>",
            "<|content_model_end_sampling|>",
            "<|message_system|>",
            "<|message_user|>",
            "<|message_model|>",
            "<|message_tool|>",
            "<|content_text|>",
            "<|end_message|>",
            "<|content_thinking|>",
            "<|content_invoke_tool_json|>",
        ]
        w.add_tokenizer_model("gpt2")
        w.add_token_list(chars + ["ab", "ac"] + special)
        w.add_token_merges(["a b", "a c"])
    else:
        w.add_tokenizer_model("none")
    physical, width, vocab = 8, 32, 64
    sliding = [i % 2 == 0 for i in range(physical)]
    meta = {
        "block_count": physical,
        "context_length": 128,
        "embedding_length": width,
        "vocab_size": vocab,
        "vocab_size_unpadded": vocab,
        "feed_forward_length": 32,
        "attention.head_count": 2,
        "attention.head_count_kv": 1,
        "attention.key_length": 32,
        "attention.value_length": 32,
        "attention.relative_length": 4,
        "attention.relative_extent": 6,
        "attention.sliding_window": 4,
        "shortconv.l_cache": 4,
        "expert_count": 4,
        "expert_used_count": 2,
        "expert_feed_forward_length": 32,
        "loop.gate_rank": 8,
        "attn_res.block_size": 2,
        "loop.start": 2,
        "loop.end": 6,
        "loop.count": passes,
        "loop.varying": True,
        "attention.layer_norm_rms_epsilon": 1e-6,
        "attention.qk_clip": 4.0,
        "attention.relative_clip": 3.0,
        "attention.log_scale_floor": 3,
        "attention.log_scale_alpha": 0.1,
        "activation.situ_beta": 4.0,
        "activation.situ_linear_beta": 25.0,
        "expert_weights_scale": 2.0,
        "final_logit_softcapping": 30.0,
    }
    for k, v in meta.items():
        if quantized and k in ("attention.head_count", "attention.head_count_kv"):
            w.add_key_value(
                "inkling2_loop." + k, [v] * physical, T.ARRAY, sub_type=T.UINT32
            )
            continue
        w.add_key_value(
            "inkling2_loop." + k,
            v,
            {int: T.UINT32, float: T.FLOAT32, bool: T.BOOL}[type(v)],
        )
    w.add_key_value(
        "inkling2_loop.attention.sliding_window_pattern",
        sliding,
        T.ARRAY,
        sub_type=T.BOOL,
    )

    def tensor(name, shape):
        rng = np.random.default_rng(crc32(name.encode()))
        arr = rng.standard_normal(shape[::-1], dtype=np.float32) / np.sqrt(
            np.float32(shape[0])
        )
        if "norm.weight" in name or "_clip.weight" in name:
            arr = 1 + 0.1 * arr
        if name.endswith("exp_probs_b.bias"):
            arr = np.array([0.4, 0.2, -0.1, -0.4], dtype=np.float32)
        # Quantized runs cover offsets within each bank of expert matrices.
        encoding = None
        if quantized:
            if name.endswith("_exps.weight") or name == "output.weight":
                encoding = Q.Q8_0
            elif name.endswith("attn_gate.weight"):
                encoding = Q.F16
            elif name.endswith("attn_v.weight"):
                encoding = Q.BF16
        if encoding is not None:
            data = quantize(arr, encoding)
            if expand:
                from gguf import dequantize

                w.add_tensor(name, dequantize(data, encoding))
            else:
                w.add_tensor(name, data, raw_dtype=encoding)
        else:
            w.add_tensor(name, arr)

    for n, shape in {
        "token_embd": (width, vocab),
        "output": (width, vocab),
        "output_norm": (width,),
        "output_res_query": (width,),
        "token_embd_norm": (width,),
    }.items():
        tensor(n + ".weight", shape)
    for i in range(physical):
        shapes = {
            "attn_q": (32, 64),
            "attn_k": (32, 32),
            "attn_v": (32, 32),
            "attn_output": (64, 32),
            "attn_gate": (32, 64),
            "attn_sink_proj": (32, 2),
            "attn_rel": (32, 8),
            "attn_q_clip": (32, 2),
            "attn_rel_clip": (4, 2),
            "attn_rel_bias": (4, 4 if sliding[i] else 6),
            "ffn_res_query": (32,),
            "ffn_norm": (32,),
            "ffn_gate_inp": (32, 4),
            "exp_probs_b": (4,),
            "ffn_gate_up_exps": (32, 64, 4),
            "ffn_down_exps": (32, 32, 4),
        }
        for n in ["attn_k_conv", "attn_v_conv", "attn_post_conv", "ffn_post_conv"]:
            shapes[n] = (4, 32)
        if i:
            shapes["attn_res_query"] = (32, 1 if sliding[i] else 3)
        if sliding[i]:
            shapes["attn_norm"] = (32,)
        else:
            for j in "qkv":
                shapes[f"attn_{j}_inp_norm"] = (32,)
            for j in "kv":
                shapes[f"attn_{j}_conv_proj"] = (32, 4)
        if 2 <= i < 6:
            shapes["loop_msg_query"] = (32, 4)
            for j in ("q", "k", "v", "ffn"):
                for tag, size in [
                    ("down", (64, 8)),
                    ("gain", (8, 32)),
                    ("forget", (8, 32)),
                ]:
                    shapes[f"loop_{j}_ctrl_{tag}"] = size
        for n, shape in shapes.items():
            tensor(
                f"blk.{i}.{n}." + ("bias" if n == "exp_probs_b" else "weight"), shape
            )
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def reference(build, model, ids, cache):
    result = subprocess.run(
        [str(build / "varloop_oracle.exe"), str(model), cache, *map(str, ids)],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return np.frombuffer(result.stdout, "<f4").reshape(len(ids), 64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, default=Path("target/n1x"))
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()
    build = args.build.resolve()
    sys.path.insert(0, str(build / "llama-src/gguf-py"))
    ids = [7, 2, 13, 21, 4, 9, 16, 18]
    with TemporaryDirectory() as temp:
        root = Path(temp)
        template = root / "model.gguf"
        exe = str(build / "ds4-varloop.exe")
        for quant in [False, True]:
            make_model(template, 3, quant)
            distinct = []
            for passes in [1, 3, 5]:
                model = root / "reference.gguf"
                # Match the native FP32 activation arithmetic using the same stored Q8 values.
                make_model(model, passes, quant, expand=True)
                expected = reference(build, model, ids, "f32")
                floor = reference(build, model, ids, "f16")
                cpu = None
                for backend in ["cpu", "cuda"] if args.cuda else ["cpu"]:
                    target = root / f"{quant}-{passes}-{backend}.f32"
                    cmd = [
                        exe,
                        "-m",
                        str(template),
                        "--tokens",
                        ",".join(map(str, ids)),
                        "--passes",
                        str(passes),
                        "--backend",
                        backend,
                        "--dump-logits",
                        str(target),
                        "-n",
                        "0",
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    actual = np.fromfile(target, "<f4").reshape(expected.shape)
                    rms = lambda a: float(np.sqrt(np.mean(a**2)))
                    error = rms(actual - expected)
                    print(
                        f"Q8={quant} passes={passes} {backend} max={np.max(np.abs(actual - expected)):.6g} rms={error:.6g} ref_cache_rms={rms(floor - expected):.6g}",
                        flush=True,
                    )
                    if cpu is None:
                        cpu = actual
                        distinct.append(actual)
                    else:
                        np.testing.assert_allclose(actual, cpu, rtol=1e-4, atol=1e-4)
                    assert error <= max(1e-4, rms(floor - expected)), (
                        error,
                        rms(floor - expected),
                    )
                    np.testing.assert_array_equal(
                        actual.argmax(axis=-1), expected.argmax(axis=-1)
                    )
            assert all(np.max(np.abs(a - b)) > 0.1 for a, b in pairwise(distinct))
        make_model(template, 3, False, tokenizer=True)
        base = [exe, "-m", str(template), "--backend", "cpu", "-n", "1"]
        for n, inputs in enumerate((["-p", "ab"], ["--tokens", "52", "--passes", "3"])):
            subprocess.run(
                [*base, *inputs, "--dump-logits", str(root / f"text{n}")],
                check=True,
                capture_output=True,
            )
        assert (root / "text0").read_bytes() == (root / "text1").read_bytes()
        for inputs in (
            ["--tokens", "64"],
            ["--tokens", "1", "--passes", "0"],
            ["--tokens", "1", "--passes", "6"],
        ):
            assert (
                subprocess.run(
                    [*base, *inputs], capture_output=True, check=False
                ).returncode
                == 1
            )


if __name__ == "__main__":
    main()
