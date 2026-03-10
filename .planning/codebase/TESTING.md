# Testing And Verification Map

## Current Testing Shape

Testing in this repository is distributed across Python comparison scripts, evaluation scripts, and SystemVerilog testbenches. There is no visible single command, CI workflow, or unified harness that runs everything end to end.

The main verification layers are:

- reference-model and preprocessing checks in `scripts/`
- memory-file comparison utilities in `scripts/`
- SystemVerilog benches in `rtl/tb/`
- PS-side bring-up examples in `sw/`

## Python Test Assets

- `requirements.txt` includes `pytest`, but the repo does not expose a conventional `tests/` directory, `pytest.ini`, or `pyproject.toml` test configuration.
- The closest Python "test" file is `scripts/mobilenet_test.py`, but it behaves like a manual smoke script:
  - it builds random parameters
  - runs `mobilenet_v1_numpy_forward(...)`
  - prints output shape and row sums
  - it does not contain `test_*` functions or assertions
- This means `pytest` is available in the environment, but automated Python unit coverage appears minimal or absent from the checked-in tree.

## Script-Based Regression Checks

Several scripts act as executable regression tools rather than formal unit tests:

- `scripts/compare_fc.py` compares `rtl/mem/fc_expected.mem` against `rtl/mem/fc_out_hw.mem` and returns non-zero on mismatch.
- `scripts/compare_fc_logits.py` performs the same role for int32 logits.
- `scripts/compare_all_layers.py` walks `layer*_out_exp.mem` and `layer*_out_hw.mem`, stopping on the first failing layer.
- `scripts/compare_int8_paths.py` compares FP32, TFLite int8, and golden int8 outputs.
- `scripts/eval_imagenet_val.py`, `scripts/eval_golden_int8.py`, and `scripts/eval_roi_tile_skip.py` are accuracy/evaluation harnesses over validation images rather than unit tests.

These tools are useful for planning because they already encode expected artifacts, file naming, and success criteria.

## Test Data And Golden Flow

- Much of the verification pipeline depends on generated `.mem` files under `rtl/mem`.
- `scripts/gen_golden_fc.py` is a central golden-model producer. Other flows call it to create `fc_expected.mem`, `fc_logits_expected.mem`, and layer outputs.
- `scripts/run_roi_golden.py` orchestrates preprocessing plus golden-output generation by chaining `scripts/image_to_input_mem.py`, `scripts/image_to_input_mem_roi.py`, and `scripts/gen_golden_fc.py` through `subprocess.run(..., check=True)`.
- `scripts/export_mobilenet_int8_mem.py` exports ROM/init data for RTL from a TFLite model and encoded quantization assumptions.
- Many verification scripts assume external assets such as:
  - `quantized_models/mobilenet_int8_ilsvrc2012_5000.tflite`
  - ImageNet validation images under `ILSVRC2012_val`
  - ImageNet devkit metadata files

Planning implication: reproducibility depends on documenting datasets and generated memory files, not just code entrypoints.

## Python Verification Style

- Python checks are largely black-box and output-oriented.
- Helpers like `read_mem8`, `read_mem32`, and `compare_files` interpret hex memory dumps and compare actual RTL output to golden output.
- Failures are reported with useful mismatch summaries such as first bad indices and max absolute difference.
- Return codes are meaningful:
  - `scripts/compare_fc.py` returns `0` on match, `1` on shape mismatch, and `2` on value mismatch
  - `scripts/compare_all_layers.py` returns `1` as soon as any layer or FC/GAP comparison fails
- Sanity gating is used before expensive evaluation. `scripts/eval_golden_int8.py` checks one known image first and aborts the full run if the prediction is wrong.

## RTL Testbench Coverage

The `rtl/tb/` directory contains focused benches for both leaf blocks and control wrappers:

- `rtl/tb/tb_line_buffer_3x3.sv`
- `rtl/tb/tb_param_cache.sv`
- `rtl/tb/tb_mobilenet_v1_axi_lite.sv`
- `rtl/tb/tb_mobilenet_v1_axi_ctrl_bram.sv`
- `rtl/tb/tb_mobilenet_v1_vivado_bram_top.sv`
- `rtl/tb/tb_mobilenet_v1_top.sv`

Observed bench patterns:

- small benches drive hand-crafted scenarios and print observable outputs, as in `rtl/tb/tb_line_buffer_3x3.sv`
- control-plane benches implement reusable `check_equal(...)` tasks and a `failures` counter
- benches typically report pass/fail via `$display("TB PASSED")` or `$display("TB FAILED failures=%0d", failures)`
- some benches inspect internal DUT hierarchy directly, for example tile-mask memory state in `rtl/tb/tb_mobilenet_v1_axi_lite.sv`
- `rtl/tb/tb_mobilenet_v1_top.sv` is more like an integration/debug bench, with many counters, state trackers, and dump-related variables for investigating long runs

This is useful but somewhat brittle: direct hierarchical inspection couples benches to implementation structure.

## Simulator And Build Signals

- Checked-in makefiles under `obj_dir/`, such as `obj_dir/Vtb_line_buffer_3x3.mk` and `obj_dir/Vtb_mobilenet_v1_top.mk`, indicate Verilator-generated build artifacts are part of the current workflow.
- I did not find a top-level Makefile, CI workflow, or documented single command to rebuild and run all benches.
- `rtl/README.md` documents architecture and memory/export assumptions, but not a canonical regression command sequence.

## Software Bring-Up Validation

- `sw/mobilenet_v1_bringup_example.c` is not a test in the strict sense; it is a hardware bring-up skeleton.
- It does still play a verification role because it exercises:
  - register programming
  - tile-mask loading
  - start/done polling
  - output visibility through a simple byte dump
- This suggests part of the intended validation strategy is board-level smoke testing after simulation.

## What Is Missing

- No visible CI automation.
- No formal Python unit-test suite around `src/` math kernels like `src/conv.py`, `src/depthwise_conv.py`, `src/pooling.py`, or `src/Fully_connected.py`.
- No property-based or randomized RTL verification beyond handwritten scenarios.
- No shared regression runner that sequences memory generation, RTL simulation, and compare scripts.
- No explicit coverage reporting for Python or RTL.
- No fixture management for external datasets or models.

## Practical Planning Guidance

- Treat `scripts/compare_*.py` and `scripts/gen_golden_fc.py` as the current regression backbone; extending them will align with existing practice.
- If adding Python tests, start with deterministic unit tests around `src/` functions because those have the least external dependency burden.
- If adding RTL regression automation, wrap existing `rtl/tb/*.sv` benches and existing `obj_dir/*.mk` flows rather than replacing them immediately.
- Preserve the `.mem` artifact contract. Many existing tools communicate through files in `rtl/mem`, so changing filenames or formats will ripple across the workflow.
- Document external prerequisites explicitly in any future phase plan: conda env from `environment.yml`, packages from `requirements.txt`, TFLite model files, and ImageNet data locations.
