#!/usr/bin/env python3
import argparse
import os
import numpy as np


def to_signed(val, bits):
    if val & (1 << (bits - 1)):
        return val - (1 << bits)
    return val


def read_mem8(path):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16)
            vals.append(to_signed(v & 0xFF, 8))
    return np.array(vals, dtype=np.int8)


def compare_files(expected, actual, max_mismatches=5):
    if not os.path.exists(expected):
        return False, f"missing expected {expected}"
    if not os.path.exists(actual):
        return False, f"missing actual {actual}"
    exp = read_mem8(expected)
    act = read_mem8(actual)
    if exp.shape != act.shape:
        return False, f"shape mismatch exp {exp.shape} act {act.shape}"
    diff = exp.astype(np.int16) - act.astype(np.int16)
    mism = np.nonzero(diff)[0]
    if mism.size == 0:
        return True, "match"
    msg = [f"mismatches {mism.size}/{exp.size}, max abs diff {int(np.max(np.abs(diff)))}"]
    for idx in mism[:max_mismatches]:
        msg.append(f"  idx {idx}: exp {int(exp[idx])} act {int(act[idx])} diff {int(diff[idx])}")
    return False, "\n".join(msg)


def main():
    parser = argparse.ArgumentParser(description="Compare per-layer outputs vs golden.")
    parser.add_argument("--mem-dir", type=str, default="rtl/mem")
    parser.add_argument("--layers", type=int, default=14)
    parser.add_argument("--max-mismatches", type=int, default=5)
    args = parser.parse_args()

    any_fail = False
    for i in range(args.layers):
        exp = os.path.join(args.mem_dir, f"layer{i}_out_exp.mem")
        act = os.path.join(args.mem_dir, f"layer{i}_out_hw.mem")
        ok, info = compare_files(exp, act, args.max_mismatches)
        status = "OK" if ok else "FAIL"
        print(f"layer{i}: {status}\n{info}")
        if not ok:
            any_fail = True
            break

    # GAP
    if not any_fail:
        exp = os.path.join(args.mem_dir, "gap_out_exp.mem")
        act = os.path.join(args.mem_dir, "gap_out_hw.mem")
        ok, info = compare_files(exp, act, args.max_mismatches)
        status = "OK" if ok else "FAIL"
        print(f"gap: {status}\n{info}")
        any_fail = any_fail or (not ok)

    # FC
    if not any_fail:
        exp = os.path.join(args.mem_dir, "fc_expected.mem")
        act = os.path.join(args.mem_dir, "fc_out_hw.mem")
        ok, info = compare_files(exp, act, args.max_mismatches)
        status = "OK" if ok else "FAIL"
        print(f"fc: {status}\n{info}")
        any_fail = any_fail or (not ok)

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
