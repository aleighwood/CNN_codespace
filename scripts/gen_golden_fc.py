#!/usr/bin/env python3
import argparse
import os
import numpy as np

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


def unflatten_kernel(flat9):
    k = np.zeros((3, 3), dtype=np.int8)
    for i, (r, c) in enumerate(ORDER):
        k[r, c] = flat9[i]
    return k


def requant_scalar(acc, mul, bias, shift, relu6_max=None, relu6_en=True):
    acc64 = np.int64(acc)
    mult = acc64 * np.int64(mul)
    scaled = mult + np.int64(bias)
    shifted = scaled >> int(shift)
    if relu6_en:
        if shifted < 0:
            shifted = 0
        if relu6_max is not None and shifted > relu6_max:
            shifted = relu6_max
    if shifted > 127:
        shifted = 127
    if shifted < -128:
        shifted = -128
    return np.int8(shifted)


def requant_vec(acc, mul, bias, shift, relu6_max=None, relu6_en=True):
    acc64 = acc.astype(np.int64)
    mul64 = mul.astype(np.int64)
    bias64 = bias.astype(np.int64)
    shift64 = shift.astype(np.int64)
    scaled = acc64 * mul64 + bias64
    shifted = scaled >> shift64
    if relu6_en:
        shifted = np.maximum(shifted, 0)
        if relu6_max is not None:
            shifted = np.minimum(shifted, relu6_max.astype(np.int64))
    shifted = np.clip(shifted, -128, 127)
    return shifted.astype(np.int8)


def conv2d_int8(x, weights, bias_acc, mul, bias_rq, shift, relu6_max, stride, pad):
    h, w, cin = x.shape
    cout = weights.shape[0]
    out_h = (h + 2 * pad - 3) // stride + 1
    out_w = (w + 2 * pad - 3) // stride + 1
    out = np.zeros((out_h, out_w, cout), dtype=np.int8)
    x_int = x.astype(np.int32)
    for oc in range(cout):
        for oh in range(out_h):
            for ow in range(out_w):
                acc = np.int32(bias_acc[oc])
                base_r = oh * stride - pad
                base_c = ow * stride - pad
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
                            acc += x_int[in_r, in_c, ic] * np.int32(k[kh, kw])
                out[oh, ow, oc] = requant_scalar(acc, mul[oc], bias_rq[oc], shift[oc], relu6_max[oc], True)
    return out


def depthwise_conv_int8(x, weights, mul, bias_rq, shift, relu6_max, stride, pad):
    h, w, cin = x.shape
    out_h = (h + 2 * pad - 3) // stride + 1
    out_w = (w + 2 * pad - 3) // stride + 1
    out = np.zeros((out_h, out_w, cin), dtype=np.int8)
    x_int = x.astype(np.int32)
    for ch in range(cin):
        k = weights[ch]
        for oh in range(out_h):
            for ow in range(out_w):
                acc = np.int32(0)
                base_r = oh * stride - pad
                base_c = ow * stride - pad
                for kh in range(3):
                    in_r = base_r + kh
                    if in_r < 0 or in_r >= h:
                        continue
                    for kw in range(3):
                        in_c = base_c + kw
                        if in_c < 0 or in_c >= w:
                            continue
                        acc += x_int[in_r, in_c, ch] * np.int32(k[kh, kw])
                out[oh, ow, ch] = requant_scalar(acc, mul[ch], bias_rq[ch], shift[ch], relu6_max[ch], True)
    return out


def pointwise_conv_int8(x, weights, bias_acc, mul, bias_rq, shift, relu6_max):
    h, w, cin = x.shape
    cout = weights.shape[0]
    x_mat = x.reshape(-1, cin).astype(np.int32)
    w_mat = weights.astype(np.int32)
    acc = x_mat @ w_mat.T
    if bias_acc is not None:
        acc = acc + bias_acc.astype(np.int32)
    acc64 = acc.astype(np.int64)
    mul64 = mul.astype(np.int64)
    bias64 = bias_rq.astype(np.int64)
    shift64 = shift.astype(np.int64)
    scaled = acc64 * mul64 + bias64
    shifted = scaled >> shift64
    shifted = np.maximum(shifted, 0)
    shifted = np.minimum(shifted, relu6_max.astype(np.int64))
    shifted = np.clip(shifted, -128, 127)
    out = shifted.astype(np.int8).reshape(h, w, cout)
    return out


def gap_int8(x, mul, bias, shift):
    acc = x.astype(np.int32).sum(axis=(0, 1))
    acc64 = acc.astype(np.int64)
    scaled = acc64 * mul.astype(np.int64) + bias.astype(np.int64)
    shifted = scaled >> shift.astype(np.int64)
    shifted = np.clip(shifted, -128, 127)
    return shifted.astype(np.int8)


def fc_int8(x, weights, mul, bias, shift):
    x_int = x.astype(np.int32)
    w_int = weights.astype(np.int32)
    acc = w_int @ x_int
    acc64 = acc.astype(np.int64)
    scaled = acc64 * mul.astype(np.int64) + bias.astype(np.int64)
    shifted = scaled >> shift.astype(np.int64)
    shifted = np.clip(shifted, -128, 127)
    return shifted.astype(np.int8)


def write_mem8(path, values):
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            v8 = int(v) & 0xFF
            f.write(f"{v8:02X}\n")


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


def main():
    parser = argparse.ArgumentParser(description="Generate random int8 input + golden FC output.")
    parser.add_argument("--mem-dir", type=str, default="rtl/mem")
    parser.add_argument("--input-shape", type=parse_shape, default="16,16,3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-mem", type=str, default="rtl/mem/input_rand.mem")
    parser.add_argument("--expected-mem", type=str, default="rtl/mem/fc_expected.mem")
    args = parser.parse_args()

    h, w, c = args.input_shape
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
    conv1_mul = read_mem_int16(os.path.join(mem, "conv1_mul.mem"))
    conv1_bias_rq = read_mem_int32(os.path.join(mem, "conv1_bias_rq.mem"))
    conv1_shift = read_mem_int32(os.path.join(mem, "conv1_shift.mem"))
    conv1_relu6 = read_mem_int8(os.path.join(mem, "conv1_relu6.mem"))

    dw_w = read_mem_int8(os.path.join(mem, "dw_weight.mem")).reshape(-1, 9)
    dw_mul = read_mem_int16(os.path.join(mem, "dw_mul.mem"))
    dw_bias = read_mem_int32(os.path.join(mem, "dw_bias.mem"))
    dw_shift = read_mem_int32(os.path.join(mem, "dw_shift.mem"))
    dw_relu6 = read_mem_int8(os.path.join(mem, "dw_relu6.mem"))

    pw_w = read_mem_int8(os.path.join(mem, "pw_weight.mem"))
    pw_bias_acc = read_mem_int32(os.path.join(mem, "pw_bias_acc.mem"))
    pw_mul = read_mem_int16(os.path.join(mem, "pw_mul.mem"))
    pw_bias_rq = read_mem_int32(os.path.join(mem, "pw_bias_rq.mem"))
    pw_shift = read_mem_int32(os.path.join(mem, "pw_shift.mem"))
    pw_relu6 = read_mem_int8(os.path.join(mem, "pw_relu6.mem"))

    gap_mul = read_mem_int16(os.path.join(mem, "gap_mul.mem"))
    gap_bias = read_mem_int32(os.path.join(mem, "gap_bias.mem"))
    gap_shift = read_mem_int32(os.path.join(mem, "gap_shift.mem"))

    fc_w = read_mem_int8(os.path.join(mem, "fc_weight.mem"))
    fc_mul = read_mem_int16(os.path.join(mem, "fc_mul.mem"))
    fc_bias = read_mem_int32(os.path.join(mem, "fc_bias.mem"))
    fc_shift = read_mem_int32(os.path.join(mem, "fc_shift.mem"))

    # Rebuild conv1 kernels.
    conv1_k = np.zeros((32, 3, 3, 3), dtype=np.int8)
    for oc in range(32):
        for ic in range(3):
            conv1_k[oc, ic] = unflatten_kernel(conv1_w[oc, ic])

    # Conv1
    x = conv2d_int8(x_q, conv1_k, conv1_bias_acc, conv1_mul, conv1_bias_rq, conv1_shift, conv1_relu6, stride=2, pad=1)
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
    for layer_id, (pw_filters, stride) in enumerate(specs, start=1):
        in_c = x.shape[2]
        # Depthwise
        dw_k = np.zeros((in_c, 3, 3), dtype=np.int8)
        for ch in range(in_c):
            dw_k[ch] = unflatten_kernel(dw_w[dw_ch_off + ch])
        x = depthwise_conv_int8(
            x,
            dw_k,
            dw_mul[dw_ch_off : dw_ch_off + in_c],
            dw_bias[dw_ch_off : dw_ch_off + in_c],
            dw_shift[dw_ch_off : dw_ch_off + in_c],
            dw_relu6[dw_ch_off : dw_ch_off + in_c],
            stride=stride,
            pad=1,
        )
        dw_ch_off += in_c

        # Pointwise
        pw_count = pw_filters * in_c
        pw_k = pw_w[pw_w_off : pw_w_off + pw_count].reshape(pw_filters, in_c)
        x = pointwise_conv_int8(
            x,
            pw_k,
            pw_bias_acc[pw_out_off : pw_out_off + pw_filters],
            pw_mul[pw_out_off : pw_out_off + pw_filters],
            pw_bias_rq[pw_out_off : pw_out_off + pw_filters],
            pw_shift[pw_out_off : pw_out_off + pw_filters],
            pw_relu6[pw_out_off : pw_out_off + pw_filters],
        )
        pw_w_off += pw_count
        pw_out_off += pw_filters
        write_tensor_chw(os.path.join(mem, f"layer{layer_id}_out_exp.mem"), x)

    # GAP
    gap_out = gap_int8(x, gap_mul[: x.shape[2]], gap_bias[: x.shape[2]], gap_shift[: x.shape[2]])
    write_mem8(os.path.join(mem, "gap_out_exp.mem"), gap_out)

    # FC
    fc_out_ch = fc_mul.shape[0]
    fc_in_ch = gap_out.shape[0]
    fc_k = fc_w.reshape(fc_out_ch, fc_in_ch)
    fc_out = fc_int8(gap_out, fc_k, fc_mul, fc_bias, fc_shift)

    write_mem8(args.expected_mem, fc_out)

    print(f"Wrote input: {args.input_mem}")
    print(f"Wrote expected FC: {args.expected_mem}")


if __name__ == "__main__":
    main()
