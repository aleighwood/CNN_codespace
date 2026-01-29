#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.MobileNet_tf import DEPTHWISE_BLOCK_SPECS, MobileNetFunctional, export_weights_to_numpy_dict


def parse_shape(value):
    parts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("input-shape must be H,W,C")
    return tuple(parts)


def fuse_conv_bn(W, gamma, beta, mean, var, eps):
    scale = gamma / np.sqrt(var + eps)
    if W.ndim == 4:
        W_fused = W * scale.reshape(1, 1, 1, -1)
    else:
        W_fused = W * scale.reshape(1, 1, -1)
    b_fused = beta - scale * mean
    return W_fused, b_fused


def quantize_per_channel(W, axis):
    W_move = np.moveaxis(W, axis, -1)
    chans = W_move.shape[-1]
    scales = np.zeros(chans, dtype=np.float32)
    W_q = np.zeros_like(W_move, dtype=np.int8)
    for c in range(chans):
        w = W_move[..., c]
        max_abs = np.max(np.abs(w))
        if max_abs == 0:
            scale = 1.0 / 127.0
        else:
            scale = max_abs / 127.0
        scales[c] = scale
        W_q[..., c] = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    W_q = np.moveaxis(W_q, -1, axis)
    return W_q, scales


def choose_mul_shift(scale, max_mul=32767, max_shift=30):
    if scale <= 0:
        return 0, 0
    # pick largest shift such that mul fits in int16
    shift = int(np.floor(np.log2(max_mul / scale))) if scale > 0 else 0
    shift = max(0, min(max_shift, shift))
    mul = int(round(scale * (1 << shift)))
    while mul > max_mul and shift > 0:
        shift -= 1
        mul = int(round(scale * (1 << shift)))
    if mul < 0:
        mul = 0
    return mul, shift


def kernel_order_indices():
    # Order matches line_buffer_3x3 window_flat index order (i=0..8)
    return [(2, 2), (2, 1), (2, 0), (1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)]


def flatten_kernel_3x3(W_3x3):
    order = kernel_order_indices()
    return [W_3x3[r, c] for (r, c) in order]


def to_hex(value, bits=32):
    mask = (1 << bits) - 1
    return f"{value & mask:0{bits // 4}X}"


def write_mem(path, values, bits=32):
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(to_hex(int(v), bits=bits))
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Export MobileNet v1 int8 .mem files (hex) for ROM init.")
    parser.add_argument("--weights", type=str, default="mobilenet_imagenet.weights.h5")
    parser.add_argument("--output-dir", type=str, default="rtl/mem")
    parser.add_argument("--input-shape", type=parse_shape, default="224,224,3")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--input-scale", type=float, default=1.0 / 127.0)
    parser.add_argument("--relu6-scale", type=float, default=6.0 / 127.0)
    args = parser.parse_args()

    model = MobileNetFunctional(input_shape=args.input_shape)
    if args.weights and os.path.exists(args.weights):
        model.load_weights(args.weights)

    params = export_weights_to_numpy_dict(model)

    os.makedirs(args.output_dir, exist_ok=True)

    s_in_conv1 = args.input_scale
    s_out_relu6 = args.relu6_scale

    # Conv1
    conv1 = params["conv1"]
    W_conv1_f, b_conv1_f = fuse_conv_bn(
        conv1["W"], conv1["gamma"], conv1["beta"], conv1["moving_mean"], conv1["moving_var"], args.eps
    )
    W_conv1_q, W_conv1_scales = quantize_per_channel(W_conv1_f, axis=3)

    conv1_weights = []
    conv1_bias_acc = []
    conv1_mul = []
    conv1_bias_rq = []
    conv1_shift = []
    conv1_relu6 = []

    for oc in range(W_conv1_q.shape[3]):
        for ic in range(W_conv1_q.shape[2]):
            kernel = W_conv1_q[:, :, ic, oc]
            conv1_weights.extend(flatten_kernel_3x3(kernel))

        scale = s_in_conv1 * W_conv1_scales[oc] / s_out_relu6
        mul, shift = choose_mul_shift(scale)
        bias_rq = int(round((b_conv1_f[oc] / s_out_relu6) * (1 << shift)))
        conv1_bias_acc.append(0)
        conv1_mul.append(mul)
        conv1_bias_rq.append(bias_rq)
        conv1_shift.append(shift)
        conv1_relu6.append(int(min(127, np.floor(6.0 / s_out_relu6))))

    # Depthwise + Pointwise blocks
    dw_weights = []
    dw_mul = []
    dw_bias = []
    dw_shift = []
    dw_relu6 = []

    pw_weights = []
    pw_bias_acc = []
    pw_mul = []
    pw_bias_rq = []
    pw_shift = []
    pw_relu6 = []

    s_in = s_out_relu6

    for idx, (pw_filters, _) in enumerate(DEPTHWISE_BLOCK_SPECS, start=1):
        block = params[f"ds_{idx}"]

        # Depthwise
        W_dw_f, b_dw_f = fuse_conv_bn(
            block["W_depthwise"],
            block["gamma_dw"],
            block["beta_dw"],
            block["moving_mean_dw"],
            block["moving_var_dw"],
            args.eps,
        )
        W_dw_q, W_dw_scales = quantize_per_channel(W_dw_f, axis=2)

        for ch in range(W_dw_q.shape[2]):
            kernel = W_dw_q[:, :, ch]
            dw_weights.extend(flatten_kernel_3x3(kernel))

            scale = s_in * W_dw_scales[ch] / s_out_relu6
            mul, shift = choose_mul_shift(scale)
            bias_rq = int(round((b_dw_f[ch] / s_out_relu6) * (1 << shift)))
            dw_mul.append(mul)
            dw_bias.append(bias_rq)
            dw_shift.append(shift)
            dw_relu6.append(int(min(127, np.floor(6.0 / s_out_relu6))))

        # Pointwise
        W_pw_f, b_pw_f = fuse_conv_bn(
            block["W_pointwise"],
            block["gamma_pw"],
            block["beta_pw"],
            block["moving_mean_pw"],
            block["moving_var_pw"],
            args.eps,
        )
        W_pw_q, W_pw_scales = quantize_per_channel(W_pw_f, axis=3)

        for oc in range(W_pw_q.shape[3]):
            for ic in range(W_pw_q.shape[2]):
                pw_weights.append(int(W_pw_q[0, 0, ic, oc]))

            scale = s_in * W_pw_scales[oc] / s_out_relu6
            mul, shift = choose_mul_shift(scale)
            bias_rq = int(round((b_pw_f[oc] / s_out_relu6) * (1 << shift)))
            pw_bias_acc.append(0)
            pw_mul.append(mul)
            pw_bias_rq.append(bias_rq)
            pw_shift.append(shift)
            pw_relu6.append(int(min(127, np.floor(6.0 / s_out_relu6))))

        s_in = s_out_relu6

    # Global average pool params (same for all channels)
    h, w, _ = args.input_shape
    h = (h + 2 - 3) // 2 + 1
    w = (w + 2 - 3) // 2 + 1
    for _, stride in DEPTHWISE_BLOCK_SPECS:
        h = (h + 2 - 3) // stride[0] + 1
        w = (w + 2 - 3) // stride[1] + 1
    gap_scale = 1.0 / (h * w)
    gap_mul, gap_shift = choose_mul_shift(gap_scale)
    gap_mul_list = [gap_mul] * 1024
    gap_bias_list = [0] * 1024
    gap_shift_list = [gap_shift] * 1024

    # FC (conv_preds) weights/params
    fc = params["fc"]
    W_fc = fc["W"]  # (1024, classes)
    b_fc = fc["b"].reshape(-1)
    W_fc_q, W_fc_scales = quantize_per_channel(W_fc, axis=1)

    s_gap_out = s_out_relu6 * gap_scale
    s_fc_out = s_gap_out

    fc_weights = []
    fc_mul = []
    fc_bias = []
    fc_shift = []

    for oc in range(W_fc_q.shape[1]):
        for ic in range(W_fc_q.shape[0]):
            fc_weights.append(int(W_fc_q[ic, oc]))
        scale = s_gap_out * W_fc_scales[oc] / s_fc_out
        mul, shift = choose_mul_shift(scale)
        bias_rq = int(round((b_fc[oc] / s_fc_out) * (1 << shift)))
        fc_mul.append(mul)
        fc_bias.append(bias_rq)
        fc_shift.append(shift)

    # Write mem files (32-bit hex per line)
    write_mem(os.path.join(args.output_dir, "conv1_weight.mem"), conv1_weights)
    write_mem(os.path.join(args.output_dir, "conv1_bias_acc.mem"), conv1_bias_acc)
    write_mem(os.path.join(args.output_dir, "conv1_mul.mem"), conv1_mul)
    write_mem(os.path.join(args.output_dir, "conv1_bias_rq.mem"), conv1_bias_rq)
    write_mem(os.path.join(args.output_dir, "conv1_shift.mem"), conv1_shift)
    write_mem(os.path.join(args.output_dir, "conv1_relu6.mem"), conv1_relu6)

    write_mem(os.path.join(args.output_dir, "dw_weight.mem"), dw_weights)
    write_mem(os.path.join(args.output_dir, "dw_mul.mem"), dw_mul)
    write_mem(os.path.join(args.output_dir, "dw_bias.mem"), dw_bias)
    write_mem(os.path.join(args.output_dir, "dw_shift.mem"), dw_shift)
    write_mem(os.path.join(args.output_dir, "dw_relu6.mem"), dw_relu6)

    write_mem(os.path.join(args.output_dir, "pw_weight.mem"), pw_weights)
    write_mem(os.path.join(args.output_dir, "pw_bias_acc.mem"), pw_bias_acc)
    write_mem(os.path.join(args.output_dir, "pw_mul.mem"), pw_mul)
    write_mem(os.path.join(args.output_dir, "pw_bias_rq.mem"), pw_bias_rq)
    write_mem(os.path.join(args.output_dir, "pw_shift.mem"), pw_shift)
    write_mem(os.path.join(args.output_dir, "pw_relu6.mem"), pw_relu6)

    write_mem(os.path.join(args.output_dir, "gap_mul.mem"), gap_mul_list)
    write_mem(os.path.join(args.output_dir, "gap_bias.mem"), gap_bias_list)
    write_mem(os.path.join(args.output_dir, "gap_shift.mem"), gap_shift_list)

    write_mem(os.path.join(args.output_dir, "fc_weight.mem"), fc_weights)
    write_mem(os.path.join(args.output_dir, "fc_mul.mem"), fc_mul)
    write_mem(os.path.join(args.output_dir, "fc_bias.mem"), fc_bias)
    write_mem(os.path.join(args.output_dir, "fc_shift.mem"), fc_shift)


if __name__ == "__main__":
    main()
