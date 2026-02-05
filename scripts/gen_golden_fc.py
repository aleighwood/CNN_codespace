#!/usr/bin/env python3
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.MobileNet_tf import DEPTHWISE_BLOCK_SPECS

ORDER = [(2, 2), (2, 1), (2, 0), (1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)]


def to_signed(val, bits):
    if val & (1 << (bits - 1)):
        return val - (1 << bits)
    return val


def read_mem(path):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals.append(int(line, 16))
    return vals


def read_mem_int8(path):
    vals = read_mem(path)
    return np.array([to_signed(v & 0xFF, 8) for v in vals], dtype=np.int8)


def read_mem_int16(path):
    vals = read_mem(path)
    return np.array([to_signed(v & 0xFFFF, 16) for v in vals], dtype=np.int16)


def read_mem_int32(path):
    vals = read_mem(path)
    return np.array([to_signed(v & 0xFFFFFFFF, 32) for v in vals], dtype=np.int32)


def read_mem_optional_int8(path, length, fill_value=0):
    if os.path.exists(path):
        return read_mem_int8(path)
    return np.full(length, fill_value, dtype=np.int8)


def read_mem_optional_int32(path, length, fill_value=0):
    if os.path.exists(path):
        return read_mem_int32(path)
    return np.full(length, fill_value, dtype=np.int32)


def unflatten_kernel(flat9):
    k = np.zeros((3, 3), dtype=np.int8)
    for i, (r, c) in enumerate(ORDER):
        k[r, c] = flat9[i]
    return k


def requant_scalar(acc, mul, bias, shift, relu6_min=None, relu6_max=None, relu6_en=True):
    acc64 = np.int64(acc)
    mult = acc64 * np.int64(mul)
    scaled = mult + np.int64(bias)
    shifted = scaled >> int(shift)
    zp_out = 0 if relu6_min is None else int(relu6_min)
    shifted = shifted + zp_out
    if relu6_en:
        min_v = 0 if relu6_min is None else int(relu6_min)
        max_v = 127 if relu6_max is None else int(relu6_max)
        if shifted < min_v:
            shifted = min_v
        if shifted > max_v:
            shifted = max_v
    if shifted > 127:
        shifted = 127
    if shifted < -128:
        shifted = -128
    return np.int8(shifted)


def requant_vec(acc, mul, bias, shift, relu6_min=None, relu6_max=None, relu6_en=True):
    acc64 = acc.astype(np.int64)
    mul64 = mul.astype(np.int64)
    bias64 = bias.astype(np.int64)
    shift64 = shift.astype(np.int64)
    scaled = acc64 * mul64 + bias64
    shifted = scaled >> shift64
    if relu6_min is not None:
        shifted = shifted + relu6_min.astype(np.int64)
    if relu6_en:
        min_v = 0 if relu6_min is None else relu6_min.astype(np.int64)
        max_v = 127 if relu6_max is None else relu6_max.astype(np.int64)
        shifted = np.maximum(shifted, min_v)
        shifted = np.minimum(shifted, max_v)
    shifted = np.clip(shifted, -128, 127)
    return shifted.astype(np.int8)


def saturating_rounding_doubling_high_mul(a, b):
    a64 = a.astype(np.int64) if isinstance(a, np.ndarray) else np.int64(a)
    b64 = b.astype(np.int64) if isinstance(b, np.ndarray) else np.int64(b)
    ab = a64 * b64
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30)) if isinstance(ab, np.ndarray) else (1 << 30 if ab >= 0 else 1 - (1 << 30))
    ab = ab + nudge
    res = ab >> 31
    if isinstance(res, np.ndarray):
        res = np.clip(res, -2**31, 2**31 - 1)
    else:
        res = max(-2**31, min(2**31 - 1, int(res)))
    return res


def rounding_divide_by_pot(x, shift):
    if shift <= 0:
        return x
    mask = (1 << shift) - 1
    if isinstance(x, np.ndarray):
        remainder = x & mask
        threshold = (mask >> 1) + (x < 0).astype(np.int64)
        return (x >> shift) + (remainder > threshold)
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> shift) + (1 if remainder > threshold else 0)


def round_away_from_zero(x):
    if x >= 0:
        return int(math.floor(x + 0.5))
    return int(math.ceil(x - 0.5))


def quantize_multiplier(real_multiplier):
    if real_multiplier == 0.0:
        return 0, 0
    if real_multiplier >= 1.0:
        if abs(real_multiplier - 1.0) < 1e-12:
            return (1 << 31) - 1, 0
        raise ValueError(f"Expected scale <= 1.0, got {real_multiplier}")
    significand, exp = math.frexp(real_multiplier)
    q = round_away_from_zero(significand * (1 << 31))
    if q == (1 << 31):
        q //= 2
        exp += 1
    right_shift = -exp
    if right_shift < 0:
        raise ValueError(f"Unexpected negative right shift for scale {real_multiplier}")
    return q, right_shift


def multiply_by_quantized_multiplier(x, q, shift):
    return rounding_divide_by_pot(saturating_rounding_doubling_high_mul(x, q), shift)


def compute_same_padding(h, w, k, stride):
    out_h = (h + stride - 1) // stride
    out_w = (w + stride - 1) // stride
    pad_h = max((out_h - 1) * stride + k - h, 0)
    pad_w = max((out_w - 1) * stride + k - w, 0)
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    return pad_top, pad_left, out_h, out_w


def conv2d_int8_q31(x, weights, bias_acc, mul_q31, shift_q31, zp_out, relu6_min, relu6_max, stride, zp_in):
    h, w, cin = x.shape
    cout = weights.shape[0]
    pad_top, pad_left, out_h, out_w = compute_same_padding(h, w, 3, stride)
    sum_w_full = weights.astype(np.int32).sum(axis=(1, 2, 3))
    out = np.zeros((out_h, out_w, cout), dtype=np.int8)
    x_int = x.astype(np.int32)
    for oc in range(cout):
        for oh in range(out_h):
            for ow in range(out_w):
                acc = np.int32(bias_acc[oc])
                sum_w_valid = np.int32(0)
                base_r = oh * stride - pad_top
                base_c = ow * stride - pad_left
                for ic in range(cin):
                    k = weights[oc, ic]
                    for kh in range(3):
                        in_r = base_r + kh
                        if in_r < 0 or in_r >= h:
                            continue
                        for kw in range(3):
                            in_c = base_c + kw
                            if in_c < 0 or in_c >= w:
                                continue
                            w_val = np.int32(k[kh, kw])
                            sum_w_valid += w_val
                            acc += x_int[in_r, in_c, ic] * w_val
                if zp_in != 0:
                    acc = acc + np.int32(zp_in) * (np.int32(sum_w_full[oc]) - sum_w_valid)
                scaled = multiply_by_quantized_multiplier(np.int64(acc), np.int64(mul_q31[oc]), int(shift_q31[oc]))
                scaled = np.int64(scaled) + np.int64(zp_out)
                min_v = int(relu6_min) if relu6_min is not None else -128
                max_v = int(relu6_max) if relu6_max is not None else 127
                if scaled < min_v:
                    scaled = min_v
                if scaled > max_v:
                    scaled = max_v
                if scaled < -128:
                    scaled = -128
                if scaled > 127:
                    scaled = 127
                out[oh, ow, oc] = np.int8(scaled)
    return out


def depthwise_conv_int8_q31(x, weights, bias_acc, mul_q31, shift_q31, zp_out, relu6_min, relu6_max, stride, zp_in):
    h, w, cin = x.shape
    pad_top, pad_left, out_h, out_w = compute_same_padding(h, w, 3, stride)
    sum_w_full = weights.astype(np.int32).sum(axis=(1, 2))
    out = np.zeros((out_h, out_w, cin), dtype=np.int8)
    x_int = x.astype(np.int32)
    for ch in range(cin):
        k = weights[ch]
        for oh in range(out_h):
            for ow in range(out_w):
                acc = np.int32(bias_acc[ch]) if bias_acc is not None else np.int32(0)
                sum_w_valid = np.int32(0)
                base_r = oh * stride - pad_top
                base_c = ow * stride - pad_left
                for kh in range(3):
                    in_r = base_r + kh
                    if in_r < 0 or in_r >= h:
                        continue
                    for kw in range(3):
                        in_c = base_c + kw
                        if in_c < 0 or in_c >= w:
                            continue
                        w_val = np.int32(k[kh, kw])
                        sum_w_valid += w_val
                        acc += x_int[in_r, in_c, ch] * w_val
                zp = zp_in[ch] if isinstance(zp_in, np.ndarray) else zp_in
                if zp != 0:
                    acc = acc + np.int32(zp) * (np.int32(sum_w_full[ch]) - sum_w_valid)
                scaled = multiply_by_quantized_multiplier(np.int64(acc), np.int64(mul_q31[ch]), int(shift_q31[ch]))
                scaled = np.int64(scaled) + np.int64(zp_out[ch] if isinstance(zp_out, np.ndarray) else zp_out)
                min_v = int(relu6_min[ch]) if relu6_min is not None else -128
                max_v = int(relu6_max[ch]) if relu6_max is not None else 127
                if scaled < min_v:
                    scaled = min_v
                if scaled > max_v:
                    scaled = max_v
                if scaled < -128:
                    scaled = -128
                if scaled > 127:
                    scaled = 127
                out[oh, ow, ch] = np.int8(scaled)
    return out


def pointwise_conv_int8_q31(x, weights, bias_acc, mul_q31, shift_q31, zp_out, relu6_min, relu6_max):
    h, w, cin = x.shape
    cout = weights.shape[0]
    x_mat = x.reshape(-1, cin).astype(np.int32)
    w_mat = weights.astype(np.int32)
    acc = x_mat @ w_mat.T
    if bias_acc is not None:
        acc = acc + bias_acc.astype(np.int32)
    acc64 = acc.astype(np.int64)
    out_vals = np.empty_like(acc64, dtype=np.int64)
    for oc in range(cout):
        scaled = multiply_by_quantized_multiplier(acc64[:, oc], np.int64(mul_q31[oc]), int(shift_q31[oc]))
        scaled = scaled + np.int64(zp_out[oc] if isinstance(zp_out, np.ndarray) else zp_out)
        min_v = int(relu6_min[oc]) if relu6_min is not None else -128
        max_v = int(relu6_max[oc]) if relu6_max is not None else 127
        scaled = np.clip(scaled, min_v, max_v)
        scaled = np.clip(scaled, -128, 127)
        out_vals[:, oc] = scaled
    out = out_vals.astype(np.int8).reshape(h, w, cout)
    return out


def gap_int8_q31(x, mul_q31, shift_q31):
    acc = x.astype(np.int32).sum(axis=(0, 1))
    acc64 = acc.astype(np.int64)
    scaled = multiply_by_quantized_multiplier(acc64, np.int64(mul_q31), int(shift_q31))
    scaled = np.clip(scaled, -128, 127)
    return scaled.astype(np.int8)


def fc_int8_q31(x, weights, mul_q31, shift_q31, zp_out):
    x_int = x.astype(np.int32)
    w_int = weights.astype(np.int32)
    acc = w_int @ x_int
    acc64 = acc.astype(np.int64)
    out_vals = np.empty_like(acc64, dtype=np.int64)
    for oc in range(acc64.shape[0]):
        scaled = multiply_by_quantized_multiplier(acc64[oc], np.int64(mul_q31[oc]), int(shift_q31[oc]))
        scaled = scaled + np.int64(zp_out)
        if scaled < -128:
            scaled = -128
        if scaled > 127:
            scaled = 127
        out_vals[oc] = scaled
    return out_vals.astype(np.int8)


def write_mem8(path, values):
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            v8 = int(v) & 0xFF
            f.write(f"{v8:02X}\n")


def write_mem32(path, values):
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            v32 = int(v) & 0xFFFFFFFF
            f.write(f"{v32:08X}\n")


def write_tensor_chw(path, x):
    h, w, c = x.shape
    vals = []
    for ch in range(c):
        for r in range(h):
            for col in range(w):
                vals.append(x[r, col, ch])
    write_mem8(path, vals)


def parse_shape(value):
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("input-shape must be H,W,C")
    return tuple(parts)


def get_scale_zp(detail):
    scale, zp = detail.get("quantization", (0.0, 0))
    if isinstance(scale, (list, tuple, np.ndarray)):
        scale = float(scale[0])
    if isinstance(zp, (list, tuple, np.ndarray)):
        zp = int(zp[0])
    return float(scale), int(zp)


def get_weight_scales(detail):
    qp = detail.get("quantization_parameters", {})
    scales = qp.get("scales", np.array([], dtype=np.float32))
    qdim = qp.get("quantized_dimension", 0)
    return scales.astype(np.float32), qdim


def build_tflite_q31_params(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    ops = interpreter._get_ops_details()
    tensors = interpreter.get_tensor_details()
    idx2 = {t["index"]: t for t in tensors}
    conv_ops = [op for op in ops if op["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D")]

    params = {}

    # Conv1
    conv1 = conv_ops[0]
    in_scale, in_zp = get_scale_zp(idx2[conv1["inputs"][0]])
    out_scale, out_zp = get_scale_zp(idx2[conv1["outputs"][0]])
    w_scales, qdim = get_weight_scales(idx2[conv1["inputs"][1]])
    if qdim != 0:
        raise SystemExit("conv1 weight qdim unexpected")
    mul = []
    shift = []
    for s in w_scales:
        q, sh = quantize_multiplier(in_scale * float(s) / out_scale)
        mul.append(q)
        shift.append(sh)
    relu6_max = round_away_from_zero(6.0 / out_scale) + out_zp
    relu6_max = max(-128, min(127, relu6_max))
    params["conv1"] = {
        "mul": np.array(mul, dtype=np.int64),
        "shift": np.array(shift, dtype=np.int32),
        "zp_in": in_zp,
        "zp_out": out_zp,
        "relu6_min": out_zp,
        "relu6_max": relu6_max,
    }

    # Depthwise + Pointwise blocks
    dw_mul = []
    dw_shift = []
    dw_zp_in = []
    dw_zp = []
    dw_relu6_min = []
    dw_relu6_max = []

    pw_mul = []
    pw_shift = []
    pw_zp_in = []
    pw_zp = []
    pw_relu6_min = []
    pw_relu6_max = []

    for idx in range(1, 1 + len(DEPTHWISE_BLOCK_SPECS)):
        dw_op = conv_ops[1 + (idx - 1) * 2]
        pw_op = conv_ops[1 + (idx - 1) * 2 + 1]

        dw_in_scale, dw_in_zp = get_scale_zp(idx2[dw_op["inputs"][0]])
        dw_out_scale, dw_out_zp = get_scale_zp(idx2[dw_op["outputs"][0]])
        dw_scales, dw_qdim = get_weight_scales(idx2[dw_op["inputs"][1]])
        if dw_qdim != 3:
            raise SystemExit("depthwise weight qdim unexpected")
        for s in dw_scales:
            q, sh = quantize_multiplier(dw_in_scale * float(s) / dw_out_scale)
            dw_mul.append(q)
            dw_shift.append(sh)
            dw_zp_in.append(dw_in_zp)
            dw_zp.append(dw_out_zp)
            relu6_max = round_away_from_zero(6.0 / dw_out_scale) + dw_out_zp
            relu6_max = max(-128, min(127, relu6_max))
            dw_relu6_min.append(dw_out_zp)
            dw_relu6_max.append(relu6_max)

        pw_in_scale, pw_in_zp = get_scale_zp(idx2[pw_op["inputs"][0]])
        pw_out_scale, pw_out_zp = get_scale_zp(idx2[pw_op["outputs"][0]])
        pw_scales, pw_qdim = get_weight_scales(idx2[pw_op["inputs"][1]])
        if pw_qdim != 0:
            raise SystemExit("pointwise weight qdim unexpected")
        for s in pw_scales:
            q, sh = quantize_multiplier(pw_in_scale * float(s) / pw_out_scale)
            pw_mul.append(q)
            pw_shift.append(sh)
            pw_zp_in.append(pw_in_zp)
            pw_zp.append(pw_out_zp)
            relu6_max = round_away_from_zero(6.0 / pw_out_scale) + pw_out_zp
            relu6_max = max(-128, min(127, relu6_max))
            pw_relu6_min.append(pw_out_zp)
            pw_relu6_max.append(relu6_max)

    params["dw"] = {
        "mul": np.array(dw_mul, dtype=np.int64),
        "shift": np.array(dw_shift, dtype=np.int32),
        "zp_in": np.array(dw_zp_in, dtype=np.int32),
        "zp_out": np.array(dw_zp, dtype=np.int32),
        "relu6_min": np.array(dw_relu6_min, dtype=np.int32),
        "relu6_max": np.array(dw_relu6_max, dtype=np.int32),
    }
    params["pw"] = {
        "mul": np.array(pw_mul, dtype=np.int64),
        "shift": np.array(pw_shift, dtype=np.int32),
        "zp_in": np.array(pw_zp_in, dtype=np.int32),
        "zp_out": np.array(pw_zp, dtype=np.int32),
        "relu6_min": np.array(pw_relu6_min, dtype=np.int32),
        "relu6_max": np.array(pw_relu6_max, dtype=np.int32),
    }

    # GAP (MEAN)
    params["gap"] = {}
    params["gap"]["mul"], params["gap"]["shift"] = 0, 0  # placeholder, set later from size

    # FC
    fc_op = conv_ops[-1]
    fc_in_scale, _ = get_scale_zp(idx2[fc_op["inputs"][0]])
    fc_out_scale, fc_out_zp = get_scale_zp(idx2[fc_op["outputs"][0]])
    fc_scales, fc_qdim = get_weight_scales(idx2[fc_op["inputs"][1]])
    if fc_qdim != 0:
        raise SystemExit("fc weight qdim unexpected")
    fc_mul = []
    fc_shift = []
    for s in fc_scales:
        q, sh = quantize_multiplier(fc_in_scale * float(s) / fc_out_scale)
        fc_mul.append(q)
        fc_shift.append(sh)
    params["fc"] = {
        "mul": np.array(fc_mul, dtype=np.int64),
        "shift": np.array(fc_shift, dtype=np.int32),
        "zp_out": fc_out_zp,
    }

    return params


def main():
    parser = argparse.ArgumentParser(description="Generate random int8 input + golden FC output.")
    parser.add_argument("--mem-dir", type=str, default="rtl/mem")
    parser.add_argument("--input-shape", type=parse_shape, default="16,16,3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-mem", type=str, default="rtl/mem/input_rand.mem")
    parser.add_argument("--input-mem-in", type=str, default="")
    parser.add_argument("--expected-mem", type=str, default="rtl/mem/fc_expected.mem")
    parser.add_argument("--expected-logits-mem", type=str, default="rtl/mem/fc_logits_expected.mem")
    parser.add_argument("--dump-dw-layer", type=int, default=-1)
    parser.add_argument("--dump-dw-mem", type=str, default="")
    parser.add_argument("--tflite", type=str, default="", help="TFLite model for Q31 quantization params.")
    parser.add_argument("--q31", action="store_true", help="Use Q31 multipliers and TFLite quantization math.")
    args = parser.parse_args()

    h, w, c = args.input_shape
    if args.input_mem_in:
        x_in = read_mem_int8(args.input_mem_in)
        if x_in.size != h * w * c:
            raise SystemExit(f"input-mem-in size {x_in.size} != expected {h*w*c}")
        # input mem is CHW
        x_q = x_in.reshape(c, h, w).transpose(1, 2, 0).astype(np.int8)
    else:
        rng = np.random.default_rng(args.seed)
        x_f = rng.uniform(-1.0, 1.0, size=(h, w, c)).astype(np.float32)
        x_q = np.clip(np.round(x_f * 127.0), -127, 127).astype(np.int8)

    # Write input in channel-major planar order.
    input_vals = []
    for ch in range(c):
        for r in range(h):
            for col in range(w):
                input_vals.append(x_q[r, col, ch])
    write_mem8(args.input_mem, input_vals)

    # Load params from mem files (matches RTL ROMs).
    mem = args.mem_dir
    conv1_w = read_mem_int8(os.path.join(mem, "conv1_weight.mem")).reshape(32, 3, 9)
    conv1_bias_acc = read_mem_int32(os.path.join(mem, "conv1_bias_acc.mem"))
    conv1_mul = read_mem_int32(os.path.join(mem, "conv1_mul.mem"))
    conv1_bias_rq = read_mem_int32(os.path.join(mem, "conv1_bias_rq.mem"))
    conv1_shift = read_mem_int32(os.path.join(mem, "conv1_shift.mem"))
    conv1_relu6 = read_mem_int8(os.path.join(mem, "conv1_relu6.mem"))
    conv1_relu6_min = read_mem_optional_int8(os.path.join(mem, "conv1_relu6_min.mem"), conv1_relu6.shape[0], 0)

    dw_w = read_mem_int8(os.path.join(mem, "dw_weight.mem")).reshape(-1, 9)
    dw_mul = read_mem_int32(os.path.join(mem, "dw_mul.mem"))
    dw_bias = read_mem_int32(os.path.join(mem, "dw_bias.mem"))
    dw_bias_acc = read_mem_optional_int32(os.path.join(mem, "dw_bias_acc.mem"), dw_bias.shape[0], 0)
    dw_shift = read_mem_int32(os.path.join(mem, "dw_shift.mem"))
    dw_relu6 = read_mem_int8(os.path.join(mem, "dw_relu6.mem"))
    dw_relu6_min = read_mem_optional_int8(os.path.join(mem, "dw_relu6_min.mem"), dw_relu6.shape[0], 0)

    pw_w = read_mem_int8(os.path.join(mem, "pw_weight.mem"))
    pw_bias_acc = read_mem_int32(os.path.join(mem, "pw_bias_acc.mem"))
    pw_mul = read_mem_int32(os.path.join(mem, "pw_mul.mem"))
    pw_bias_rq = read_mem_int32(os.path.join(mem, "pw_bias_rq.mem"))
    pw_shift = read_mem_int32(os.path.join(mem, "pw_shift.mem"))
    pw_relu6 = read_mem_int8(os.path.join(mem, "pw_relu6.mem"))
    pw_relu6_min = read_mem_optional_int8(os.path.join(mem, "pw_relu6_min.mem"), pw_relu6.shape[0], 0)

    gap_mul = read_mem_int32(os.path.join(mem, "gap_mul.mem"))
    gap_bias = read_mem_int32(os.path.join(mem, "gap_bias.mem"))
    gap_shift = read_mem_int32(os.path.join(mem, "gap_shift.mem"))

    fc_w = read_mem_int8(os.path.join(mem, "fc_weight.mem"))
    fc_mul = read_mem_int32(os.path.join(mem, "fc_mul.mem"))
    fc_bias = read_mem_int32(os.path.join(mem, "fc_bias.mem"))
    fc_bias_acc = read_mem_optional_int32(os.path.join(mem, "fc_bias_acc.mem"), fc_bias.shape[0], 0)
    fc_shift = read_mem_int32(os.path.join(mem, "fc_shift.mem"))
    fc_zp = read_mem_optional_int32(os.path.join(mem, "fc_zp.mem"), fc_bias.shape[0], 0)

    # Rebuild conv1 kernels.
    conv1_k = np.zeros((32, 3, 3, 3), dtype=np.int8)
    for oc in range(32):
        for ic in range(3):
            conv1_k[oc, ic] = unflatten_kernel(conv1_w[oc, ic])

    use_q31 = bool(args.q31 and args.tflite)
    if use_q31:
        qparams = build_tflite_q31_params(args.tflite)
    else:
        qparams = None

    # Conv1
    if use_q31:
        x = conv2d_int8_q31(
            x_q,
            conv1_k,
            conv1_bias_acc,
            qparams["conv1"]["mul"],
            qparams["conv1"]["shift"],
            qparams["conv1"]["zp_out"],
            qparams["conv1"]["relu6_min"],
            qparams["conv1"]["relu6_max"],
            stride=2,
            zp_in=qparams["conv1"]["zp_in"],
        )
    else:
        x = conv2d_int8(
            x_q,
            conv1_k,
            conv1_bias_acc,
            conv1_mul,
            conv1_bias_rq,
            conv1_shift,
            conv1_relu6_min,
            conv1_relu6,
            stride=2,
            pad=1,
        )
    write_tensor_chw(os.path.join(mem, "layer0_out_exp.mem"), x)

    # Depthwise + Pointwise blocks (MobileNet v1)
    specs = [
        (64, 1),
        (128, 2),
        (128, 1),
        (256, 2),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (512, 1),
        (1024, 2),
        (1024, 1),
    ]

    dw_ch_off = 0
    pw_w_off = 0
    pw_out_off = 0
    dw_dump_path = ""
    if args.dump_dw_layer >= 0:
        if args.dump_dw_mem:
            dw_dump_path = args.dump_dw_mem
        else:
            dw_dump_path = os.path.join(mem, f"layer{args.dump_dw_layer}_dw_exp.mem")

    for layer_id, (pw_filters, stride) in enumerate(specs, start=1):
        in_c = x.shape[2]
        # Depthwise
        dw_k = np.zeros((in_c, 3, 3), dtype=np.int8)
        for ch in range(in_c):
            dw_k[ch] = unflatten_kernel(dw_w[dw_ch_off + ch])
        if use_q31:
            x = depthwise_conv_int8_q31(
                x,
                dw_k,
                dw_bias_acc[dw_ch_off : dw_ch_off + in_c],
                qparams["dw"]["mul"][dw_ch_off : dw_ch_off + in_c],
                qparams["dw"]["shift"][dw_ch_off : dw_ch_off + in_c],
                qparams["dw"]["zp_out"][dw_ch_off : dw_ch_off + in_c],
                qparams["dw"]["relu6_min"][dw_ch_off : dw_ch_off + in_c],
                qparams["dw"]["relu6_max"][dw_ch_off : dw_ch_off + in_c],
                stride=stride,
                zp_in=qparams["dw"]["zp_in"][dw_ch_off : dw_ch_off + in_c],
            )
        else:
            x = depthwise_conv_int8(
                x,
                dw_k,
                dw_bias_acc[dw_ch_off : dw_ch_off + in_c],
                dw_mul[dw_ch_off : dw_ch_off + in_c],
                dw_bias[dw_ch_off : dw_ch_off + in_c],
                dw_shift[dw_ch_off : dw_ch_off + in_c],
                dw_relu6_min[dw_ch_off : dw_ch_off + in_c],
                dw_relu6[dw_ch_off : dw_ch_off + in_c],
                stride=stride,
                pad=1,
            )
        if dw_dump_path and layer_id == args.dump_dw_layer:
            write_tensor_chw(dw_dump_path, x)
        dw_ch_off += in_c

        # Pointwise
        pw_count = pw_filters * in_c
        pw_k = pw_w[pw_w_off : pw_w_off + pw_count].reshape(pw_filters, in_c)
        if use_q31:
            x = pointwise_conv_int8_q31(
                x,
                pw_k,
                pw_bias_acc[pw_out_off : pw_out_off + pw_filters],
                qparams["pw"]["mul"][pw_out_off : pw_out_off + pw_filters],
                qparams["pw"]["shift"][pw_out_off : pw_out_off + pw_filters],
                qparams["pw"]["zp_out"][pw_out_off : pw_out_off + pw_filters],
                qparams["pw"]["relu6_min"][pw_out_off : pw_out_off + pw_filters],
                qparams["pw"]["relu6_max"][pw_out_off : pw_out_off + pw_filters],
            )
        else:
            x = pointwise_conv_int8(
                x,
                pw_k,
                pw_bias_acc[pw_out_off : pw_out_off + pw_filters],
                pw_mul[pw_out_off : pw_out_off + pw_filters],
                pw_bias_rq[pw_out_off : pw_out_off + pw_filters],
                pw_shift[pw_out_off : pw_out_off + pw_filters],
                pw_relu6_min[pw_out_off : pw_out_off + pw_filters],
                pw_relu6[pw_out_off : pw_out_off + pw_filters],
            )
        pw_w_off += pw_count
        pw_out_off += pw_filters
        write_tensor_chw(os.path.join(mem, f"layer{layer_id}_out_exp.mem"), x)

    # GAP
    if use_q31:
        gap_scale = 1.0 / (x.shape[0] * x.shape[1])
        qparams["gap"]["mul"], qparams["gap"]["shift"] = quantize_multiplier(gap_scale)
        gap_out = gap_int8_q31(x, qparams["gap"]["mul"], qparams["gap"]["shift"])
    else:
        gap_out = gap_int8(x, gap_mul[: x.shape[2]], gap_bias[: x.shape[2]], gap_shift[: x.shape[2]])
    write_mem8(os.path.join(mem, "gap_out_exp.mem"), gap_out)

    # FC
    fc_out_ch = fc_mul.shape[0]
    fc_in_ch = gap_out.shape[0]
    fc_k = fc_w.reshape(fc_out_ch, fc_in_ch)
    if use_q31:
        acc32 = (fc_k.astype(np.int32) @ gap_out.astype(np.int32)).astype(np.int64)
        if fc_bias_acc is not None:
            acc32 = acc32 + fc_bias_acc.astype(np.int64)
        logits = np.empty_like(acc32, dtype=np.int64)
        for oc in range(fc_out_ch):
            scaled = multiply_by_quantized_multiplier(
                acc32[oc], np.int64(qparams["fc"]["mul"][oc]), int(qparams["fc"]["shift"][oc])
            )
            scaled = scaled + np.int64(qparams["fc"]["zp_out"])
            logits[oc] = scaled
        fc_out = np.clip(logits, -128, 127).astype(np.int8)
    else:
        # FC: int32 logits before int8 clip
        acc32 = (fc_k.astype(np.int32) @ gap_out.astype(np.int32)).astype(np.int64)
        if fc_bias_acc is not None:
            acc32 = acc32 + fc_bias_acc.astype(np.int64)
        logits = (acc32 * fc_mul.astype(np.int64) + fc_bias.astype(np.int64)) >> fc_shift.astype(np.int64)
        if fc_zp is not None:
            logits = logits + fc_zp.astype(np.int64)
        fc_out = np.clip(logits, -128, 127).astype(np.int8)

    write_mem8(args.expected_mem, fc_out)
    write_mem32(args.expected_logits_mem, logits.astype(np.int32))

    print(f"Wrote input: {args.input_mem}")
    print(f"Wrote expected FC: {args.expected_mem}")
    print(f"Wrote expected FC logits: {args.expected_logits_mem}")


if __name__ == "__main__":
    main()
