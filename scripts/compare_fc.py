#!/usr/bin/env python3
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Compare RTL FC output vs golden.")
    parser.add_argument("--expected", type=str, default="rtl/mem/fc_expected.mem")
    parser.add_argument("--actual", type=str, default="rtl/mem/fc_out_hw.mem")
    parser.add_argument("--max-mismatches", type=int, default=10)
    args = parser.parse_args()

    exp = read_mem8(args.expected)
    act = read_mem8(args.actual)

    if exp.shape != act.shape:
        print(f"Shape mismatch: expected {exp.shape}, actual {act.shape}")
        return 1

    diff = exp.astype(np.int16) - act.astype(np.int16)
    mism = np.nonzero(diff)[0]
    if mism.size == 0:
        print("FC outputs match.")
        print(f"Top-1 index: {int(np.argmax(act))}")
        return 0

    print(f"Mismatches: {mism.size} / {exp.size}")
    for idx in mism[: args.max_mismatches]:
        print(f"idx {idx}: expected {int(exp[idx])} actual {int(act[idx])} diff {int(diff[idx])}")
    print(f"Max abs diff: {int(np.max(np.abs(diff)))}")
    print(f"Top-1 expected: {int(np.argmax(exp))} actual: {int(np.argmax(act))}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
