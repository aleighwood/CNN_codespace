#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

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


def choose_mul_shift(scale):
    if scale <= 0:
        return 0, 0
    if scale >= 1.0:
        if abs(scale - 1.0) < 1e-12:
            return (1 << 31) - 1, 0
        raise SystemExit(f"Expected scale <= 1.0 for Q31 requant, got {scale}")
    q, exp = np.frexp(scale)
    q_fixed = round_away_from_zero(q * (1 << 31))
    if q_fixed == (1 << 31):
        q_fixed //= 2
        exp += 1
    right_shift = -exp
    if right_shift < 0:
        raise SystemExit(f"Unexpected negative right shift for scale {scale}")
    return q_fixed, right_shift


def round_away_from_zero(x):
    if x >= 0:
        return int(np.floor(x + 0.5))
    return int(np.ceil(x - 0.5))


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


def load_tflite(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    ops = interpreter._get_ops_details()
    tensors = interpreter.get_tensor_details()
    idx2det = {t["index"]: t for t in tensors}
    conv_ops = [op for op in ops if op["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D")]
    return interpreter, conv_ops, idx2det


def get_qparams(tensor_detail):
    qp = tensor_detail.get("quantization_parameters", {})
    scales = qp.get("scales", np.array([], dtype=np.float32))
    zero_points = qp.get("zero_points", np.array([], dtype=np.int64))
    qdim = qp.get("quantized_dimension", 0)
    return scales, zero_points, qdim


def get_scale_zp(tensor_detail):
    scale, zp = tensor_detail.get("quantization", (0.0, 0))
    if isinstance(scale, (list, tuple, np.ndarray)):
        if len(scale) != 1:
            raise SystemExit(f"Expected per-tensor scale, got {scale}")
        scale = scale[0]
    if isinstance(zp, (list, tuple, np.ndarray)):
        if len(zp) != 1:
            raise SystemExit(f"Expected per-tensor zero point, got {zp}")
        zp = zp[0]
    return float(scale), int(zp)


def main():
    parser = argparse.ArgumentParser(description="Export MobileNet v1 int8 .mem files (hex) for ROM init.")
    parser.add_argument("--weights", type=str, default="mobilenet_imagenet.weights.h5")
    parser.add_argument("--output-dir", type=str, default="rtl/mem")
    parser.add_argument("--input-shape", type=parse_shape, default="224,224,3")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--input-scale", type=float, default=1.0 / 127.0)
    parser.add_argument("--relu6-scale", type=float, default=6.0 / 127.0)
    parser.add_argument("--tflite", type=str, default="quantized_models/mobilenet_int8_ilsvrc2012_5000.tflite")
    parser.add_argument("--use-tflite-weights", action="store_true", help="Use weights/biases from TFLite model.")
    args = parser.parse_args()

    if args.use_tflite_weights:
        if not os.path.exists(args.tflite):
            raise SystemExit(f"TFLite model not found: {args.tflite}")
        interpreter, conv_ops, idx2det = load_tflite(args.tflite)
        expected_ops = 1 + len(DEPTHWISE_BLOCK_SPECS) * 2 + 1
        if len(conv_ops) != expected_ops:
            raise SystemExit(f"Expected {expected_ops} conv ops, found {len(conv_ops)} in {args.tflite}")
    else:
        model = MobileNetFunctional(input_shape=args.input_shape)
        if args.weights and os.path.exists(args.weights):
            model.load_weights(args.weights)
        params = export_weights_to_numpy_dict(model)

    os.makedirs(args.output_dir, exist_ok=True)

    s_in_conv1 = args.input_scale
    s_out_relu6 = args.relu6_scale

    # Conv1
    if args.use_tflite_weights:
        conv1_op = conv_ops[0]
        w_idx = conv1_op["inputs"][1]
        b_idx = conv1_op["inputs"][2]
        w_det = idx2det[w_idx]
        in_det = idx2det[conv1_op["inputs"][0]]
        W_conv1_q = interpreter.get_tensor(w_idx).astype(np.int8)
        bias_q = interpreter.get_tensor(b_idx).astype(np.int32)
        w_scales, _, w_qdim = get_qparams(w_det)
        if w_qdim != 0:
            raise SystemExit(f"Unexpected conv1 weight qdim {w_qdim}")
        in_scales, _, _ = get_qparams(in_det)
        if in_scales.size == 0:
            raise SystemExit("Conv1 input has no quantization scale.")
        bias_q_int = bias_q.astype(np.int32)
        bias_real = bias_q_int.astype(np.float32) * (in_scales[0] * w_scales)
        W_conv1_scales = w_scales.astype(np.float32)
    else:
        conv1 = params["conv1"]
        W_conv1_f, b_conv1_f = fuse_conv_bn(
            conv1["W"], conv1["gamma"], conv1["beta"], conv1["moving_mean"], conv1["moving_var"], args.eps
        )
        W_conv1_q, W_conv1_scales = quantize_per_channel(W_conv1_f, axis=3)
        bias_real = b_conv1_f.astype(np.float32)

    conv1_weights = []
    conv1_bias_acc = []
    conv1_mul = []
    conv1_bias_rq = []
    conv1_shift = []
    conv1_relu6 = []
    conv1_relu6_min = []

    if args.use_tflite_weights:
        out_ch, _, _, in_ch = W_conv1_q.shape
        out_det = idx2det[conv1_op["outputs"][0]]
        s_in_conv1, zp_in = get_scale_zp(in_det)
        s_out, zp_out = get_scale_zp(out_det)
        for oc in range(out_ch):
            for ic in range(in_ch):
                kernel = W_conv1_q[oc, :, :, ic]
                conv1_weights.extend(flatten_kernel_3x3(kernel))

            sum_w = int(np.sum(W_conv1_q[oc]).astype(np.int64))
            bias_acc = int(bias_q_int[oc]) - int(zp_in) * sum_w
            conv1_bias_acc.append(bias_acc)

            scale = float(s_in_conv1) * float(W_conv1_scales[oc]) / float(s_out)
            mul, shift = choose_mul_shift(scale)
            bias_rq = 0
            conv1_mul.append(mul)
            conv1_bias_rq.append(bias_rq)
            conv1_shift.append(shift)
            relu6_max = round_away_from_zero(6.0 / s_out) + int(zp_out)
            relu6_max = max(-128, min(127, relu6_max))
            conv1_relu6.append(relu6_max)
            conv1_relu6_min.append(int(max(-128, min(127, zp_out))))
    else:
        for oc in range(W_conv1_q.shape[3]):
            for ic in range(W_conv1_q.shape[2]):
                kernel = W_conv1_q[:, :, ic, oc]
                conv1_weights.extend(flatten_kernel_3x3(kernel))

            scale = s_in_conv1 * W_conv1_scales[oc] / s_out_relu6
            mul, shift = choose_mul_shift(scale)
            bias_rq = int(round((bias_real[oc] / s_out_relu6) * (1 << shift)))
            conv1_bias_acc.append(0)
            conv1_mul.append(mul)
            conv1_bias_rq.append(bias_rq)
            conv1_shift.append(shift)
            conv1_relu6.append(int(min(127, np.floor(6.0 / s_out_relu6))))
            conv1_relu6_min.append(0)

    # Depthwise + Pointwise blocks
    dw_weights = []
    dw_mul = []
    dw_bias = []
    dw_bias_acc = []
    dw_shift = []
    dw_relu6 = []
    dw_relu6_min = []

    pw_weights = []
    pw_bias_acc = []
    pw_mul = []
    pw_bias_rq = []
    pw_shift = []
    pw_relu6 = []
    pw_relu6_min = []

    s_in = s_out_relu6

    for idx, (pw_filters, _) in enumerate(DEPTHWISE_BLOCK_SPECS, start=1):
        if args.use_tflite_weights:
            dw_op = conv_ops[1 + (idx - 1) * 2]
            pw_op = conv_ops[1 + (idx - 1) * 2 + 1]

            # Depthwise
            dw_w_idx = dw_op["inputs"][1]
            dw_b_idx = dw_op["inputs"][2]
            dw_in_det = idx2det[dw_op["inputs"][0]]
            dw_out_det = idx2det[dw_op["outputs"][0]]
            dw_w_det = idx2det[dw_w_idx]
            W_dw_q = interpreter.get_tensor(dw_w_idx).astype(np.int8)  # (1,3,3,in_ch)
            dw_bias_q = interpreter.get_tensor(dw_b_idx).astype(np.int32)
            dw_scales, _, dw_qdim = get_qparams(dw_w_det)
            if dw_qdim != 3:
                raise SystemExit(f"Unexpected depthwise weight qdim {dw_qdim}")
            dw_in_scale, dw_in_zp = get_scale_zp(dw_in_det)
            dw_out_scale, dw_out_zp = get_scale_zp(dw_out_det)

            _, _, _, in_ch = W_dw_q.shape
            for ch in range(in_ch):
                kernel = W_dw_q[0, :, :, ch]
                dw_weights.extend(flatten_kernel_3x3(kernel))

                sum_w = int(np.sum(W_dw_q[0, :, :, ch]).astype(np.int64))
                bias_acc = int(dw_bias_q[ch]) - int(dw_in_zp) * sum_w
                dw_bias_acc.append(bias_acc)

                scale = float(dw_in_scale) * float(dw_scales[ch]) / float(dw_out_scale)
                mul, shift = choose_mul_shift(scale)
                bias_rq = 0
                dw_mul.append(mul)
                dw_bias.append(bias_rq)
                dw_shift.append(shift)
                relu6_max = round_away_from_zero(6.0 / dw_out_scale) + int(dw_out_zp)
                relu6_max = max(-128, min(127, relu6_max))
                dw_relu6.append(relu6_max)
                dw_relu6_min.append(int(max(-128, min(127, dw_out_zp))))

            # Pointwise
            pw_w_idx = pw_op["inputs"][1]
            pw_b_idx = pw_op["inputs"][2]
            pw_in_det = idx2det[pw_op["inputs"][0]]
            pw_out_det = idx2det[pw_op["outputs"][0]]
            pw_w_det = idx2det[pw_w_idx]
            W_pw_q = interpreter.get_tensor(pw_w_idx).astype(np.int8)  # (out,1,1,in)
            pw_bias_q = interpreter.get_tensor(pw_b_idx).astype(np.int32)
            pw_scales, _, pw_qdim = get_qparams(pw_w_det)
            if pw_qdim != 0:
                raise SystemExit(f"Unexpected pointwise weight qdim {pw_qdim}")
            pw_in_scale, pw_in_zp = get_scale_zp(pw_in_det)
            pw_out_scale, pw_out_zp = get_scale_zp(pw_out_det)

            out_ch, _, _, in_ch = W_pw_q.shape
            for oc in range(out_ch):
                for ic in range(in_ch):
                    pw_weights.append(int(W_pw_q[oc, 0, 0, ic]))

                sum_w = int(np.sum(W_pw_q[oc, 0, 0, :]).astype(np.int64))
                bias_acc = int(pw_bias_q[oc]) - int(pw_in_zp) * sum_w
                pw_bias_acc.append(bias_acc)

                scale = float(pw_in_scale) * float(pw_scales[oc]) / float(pw_out_scale)
                mul, shift = choose_mul_shift(scale)
                bias_rq = 0
                pw_mul.append(mul)
                pw_bias_rq.append(bias_rq)
                pw_shift.append(shift)
                relu6_max = round_away_from_zero(6.0 / pw_out_scale) + int(pw_out_zp)
                relu6_max = max(-128, min(127, relu6_max))
                pw_relu6.append(relu6_max)
                pw_relu6_min.append(int(max(-128, min(127, pw_out_zp))))
        else:
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
                dw_bias_acc.append(0)
                dw_shift.append(shift)
                dw_relu6.append(int(min(127, np.floor(6.0 / s_out_relu6))))
                dw_relu6_min.append(0)

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
                pw_relu6_min.append(0)

        if not args.use_tflite_weights:
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
    if args.use_tflite_weights:
        fc_op = conv_ops[-1]
        fc_w_idx = fc_op["inputs"][1]
        fc_b_idx = fc_op["inputs"][2]
        fc_in_det = idx2det[fc_op["inputs"][0]]
        fc_out_det = idx2det[fc_op["outputs"][0]]
        fc_w_det = idx2det[fc_w_idx]
        W_fc_q = interpreter.get_tensor(fc_w_idx).astype(np.int8)  # (out,1,1,in)
        fc_bias_q = interpreter.get_tensor(fc_b_idx).astype(np.int32)
        fc_scales, _, fc_qdim = get_qparams(fc_w_det)
        if fc_qdim != 0:
            raise SystemExit(f"Unexpected FC weight qdim {fc_qdim}")
        fc_in_scale, fc_in_zp = get_scale_zp(fc_in_det)
        fc_out_scale, fc_out_zp = get_scale_zp(fc_out_det)
        out_ch, _, _, in_ch = W_fc_q.shape
        W_fc_q = W_fc_q[:, 0, 0, :].reshape(out_ch, in_ch).T
        W_fc_scales = fc_scales.astype(np.float32)
        b_fc = fc_bias_q.astype(np.int32)
        s_gap_out = fc_in_scale
        s_fc_out = fc_out_scale
        fc_in_zp_val = fc_in_zp
        fc_out_zp_val = fc_out_zp
    else:
        fc = params["fc"]
        W_fc = fc["W"]  # (1024, classes)
        b_fc = fc["b"].reshape(-1)
        W_fc_q, W_fc_scales = quantize_per_channel(W_fc, axis=1)

    # Keep GAP output scale the same as input scale (avg is done in integer domain)
    if not args.use_tflite_weights:
        s_gap_out = s_out_relu6
        s_fc_out = s_gap_out

    fc_weights = []
    fc_mul = []
    fc_bias = []
    fc_bias_acc = []
    fc_shift = []
    fc_zp = []

    for oc in range(W_fc_q.shape[1]):
        for ic in range(W_fc_q.shape[0]):
            fc_weights.append(int(W_fc_q[ic, oc]))
        scale = s_gap_out * W_fc_scales[oc] / s_fc_out
        mul, shift = choose_mul_shift(scale)
        if args.use_tflite_weights:
            sum_w = int(np.sum(W_fc_q[:, oc]).astype(np.int64))
            bias_acc = int(b_fc[oc]) - int(fc_in_zp_val) * sum_w
            fc_bias_acc.append(bias_acc)
            bias_rq = 0
            fc_zp.append(int(fc_out_zp_val))
        else:
            fc_bias_acc.append(0)
            bias_rq = int(round((b_fc[oc] / s_fc_out) * (1 << shift)))
            fc_zp.append(0)
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
    write_mem(os.path.join(args.output_dir, "conv1_relu6_min.mem"), conv1_relu6_min)

    write_mem(os.path.join(args.output_dir, "dw_weight.mem"), dw_weights)
    write_mem(os.path.join(args.output_dir, "dw_mul.mem"), dw_mul)
    write_mem(os.path.join(args.output_dir, "dw_bias.mem"), dw_bias)
    write_mem(os.path.join(args.output_dir, "dw_bias_acc.mem"), dw_bias_acc)
    write_mem(os.path.join(args.output_dir, "dw_shift.mem"), dw_shift)
    write_mem(os.path.join(args.output_dir, "dw_relu6.mem"), dw_relu6)
    write_mem(os.path.join(args.output_dir, "dw_relu6_min.mem"), dw_relu6_min)

    write_mem(os.path.join(args.output_dir, "pw_weight.mem"), pw_weights)
    write_mem(os.path.join(args.output_dir, "pw_bias_acc.mem"), pw_bias_acc)
    write_mem(os.path.join(args.output_dir, "pw_mul.mem"), pw_mul)
    write_mem(os.path.join(args.output_dir, "pw_bias_rq.mem"), pw_bias_rq)
    write_mem(os.path.join(args.output_dir, "pw_shift.mem"), pw_shift)
    write_mem(os.path.join(args.output_dir, "pw_relu6.mem"), pw_relu6)
    write_mem(os.path.join(args.output_dir, "pw_relu6_min.mem"), pw_relu6_min)

    write_mem(os.path.join(args.output_dir, "gap_mul.mem"), gap_mul_list)
    write_mem(os.path.join(args.output_dir, "gap_bias.mem"), gap_bias_list)
    write_mem(os.path.join(args.output_dir, "gap_shift.mem"), gap_shift_list)

    write_mem(os.path.join(args.output_dir, "fc_weight.mem"), fc_weights)
    write_mem(os.path.join(args.output_dir, "fc_mul.mem"), fc_mul)
    write_mem(os.path.join(args.output_dir, "fc_bias.mem"), fc_bias)
    write_mem(os.path.join(args.output_dir, "fc_bias_acc.mem"), fc_bias_acc)
    write_mem(os.path.join(args.output_dir, "fc_shift.mem"), fc_shift)
    write_mem(os.path.join(args.output_dir, "fc_zp.mem"), fc_zp)


if __name__ == "__main__":
    main()
