# Architecture Map

## Repository Role
This repository is a MobileNet v1 research and implementation workspace, not a single deployable app.
Its main layers are:
- a Python reference model and quantization/export toolchain in `src/` and `scripts/`
- an int8 tiled accelerator in `rtl/`
- PS-side bring-up helpers in `sw/`
- generated assets and datasets such as `rtl/mem/`, `quantized_models/`, `ILSVRC2012_val/`, and `processed_frames/`

The repo is organized around moving one model definition through several representations: TensorFlow/Keras, NumPy golden logic, exported `.mem` parameter files, RTL simulation, then AXI/BRAM board integration.

## Primary Execution Flow
## 1. Define or load the model
`src/MobileNet_tf.py` builds the reference Keras network with `MobileNetFunctional()` and can export weights into the dictionary format used by NumPy inference.

## 2. Generate quantized or exported artifacts
The main conversion/export path is driven by scripts:
- `scripts/quantize_from_imagenet_val.py` creates `.tflite` models in `quantized_models/`
- `scripts/export_mobilenet_int8_mem.py` converts model parameters into ROM/RAM init files under `rtl/mem/`
- `scripts/build_imagenet_npz.py` prepares dataset snapshots used by benchmarking scripts

## 3. Prepare inputs, masks, and golden outputs
Runtime assets are prepared separately from the model:
- `scripts/image_to_input_mem.py` and `scripts/image_to_input_mem_roi.py` write CHW int8 input memories
- `scripts/object_crop.py` and ROI utilities write masks into `processed_frames/` and `selected_frames/`
- `scripts/gen_tile_mask_mem.py` writes tile-skip masks into `rtl/mem/tile_mask.mem`
- `scripts/gen_golden_fc.py` generates expected layer, GAP, and FC outputs consumed by RTL checking

## 4. Run RTL simulation or synthesis-oriented integration
The core hardware path starts at `rtl/blocks/mobilenet_v1_top.sv`.
That top wires:
- `rtl/blocks/mobilenet_v1_ctrl.sv` for layer sequencing and ping-pong feature-map control
- `rtl/blocks/conv1_tile_runner.sv` for the initial 3x3 stride-2 layer
- `rtl/blocks/dws_tile_runner.sv` for depthwise + pointwise tiled execution
- `rtl/blocks/gap_runner.sv` and `rtl/blocks/fc_runner.sv` for the classifier head
- `rtl/blocks/mobilenet_v1_param_cache.sv` for weight and quant-parameter lookup

## 5. Expose the core to software
Board-facing wrappers add registers and host memory access:
- `rtl/blocks/mobilenet_v1_reg_shell.sv` defines the bus-neutral register map
- `rtl/blocks/mobilenet_v1_axi_lite.sv` exposes the legacy AXI-Lite-only control path
- `rtl/blocks/mobilenet_v1_axi_ctrl_bram.sv` is the preferred split-control design: AXI-Lite for registers plus a BRAM-style feature-map port
- `rtl/blocks/mobilenet_v1_htg_vsl5_top.sv` and `rtl/blocks/mobilenet_v1_vivado_bram_top.sv` are synthesis/package entry points

## 6. Drive the accelerator from software
`sw/mobilenet_v1_ctrl_bram_regs.h` and `sw/mobilenet_v1_ctrl_bram_xil.h` mirror the hardware register map.
`sw/mobilenet_v1_vitis_bringup_example.c` shows the intended flow: load feature-map memory, load tile mask, set input shape, start inference, poll done, then inspect outputs.

## Major Subsystems
## Python reference layer
`src/MobileNet_tf.py` and `src/MobileNet_np.py` define the canonical architecture.
The same MobileNet topology is encoded twice:
- Keras for training-compatible structure and weight loading
- NumPy for transparent golden-model math and debug-friendly forward passes

The lightweight math helpers in `src/conv.py`, `src/depthwise_conv.py`, `src/batchnorm.py`, `src/pooling.py`, `src/Fully_connected.py`, and `src/ReLU6.py` are not a framework; they are small inference kernels used by the NumPy model and ad hoc validation scripts.

## Script pipeline layer
The scripts directory acts like an operator toolbox rather than a library package.
Patterns visible in `scripts/*.py`:
- most files expose `parse_args()` plus `main()`
- many scripts add `REPO_ROOT` to `sys.path` so they can import `src.*` directly
- outputs are persisted as files, usually under `rtl/mem/`, `quantized_models/`, `processed_frames/`, or dataset roots

There are several script families:
- export and quantization: `scripts/export_mobilenet_int8_mem.py`, `scripts/quantize_from_imagenet_val.py`, `scripts/quantization_benchmark.py`
- evaluation and comparison: `scripts/eval_imagenet_val.py`, `scripts/eval_golden_int8.py`, `scripts/compare_all_layers.py`, `scripts/compare_int8_paths.py`
- ROI and tile skipping: `scripts/eval_roi_tile_skip.py`, `scripts/run_roi_golden.py`, `scripts/sweep_bitmap.py`, `scripts/roi_compute_savings.py`
- preprocessing and visualization: `scripts/object_crop.py`, `scripts/image_to_input_mem.py`, `scripts/roi_visualize.py`, `scripts/roi_tile_overlay.py`

## RTL subsystem decomposition
The RTL is split by abstraction level rather than by synthesis target.

Common compute/storage primitives in `rtl/common/`:
- `rtl/common/line_buffer_3x3.sv`
- `rtl/common/dw_conv_3x3.sv`
- `rtl/common/conv3x3_mac_vec.sv`
- `rtl/common/requant_q31.sv`
- `rtl/common/tile_buf.sv`
- `rtl/common/dual_port_ram_async.sv`

Pipeline/control blocks in `rtl/blocks/`:
- data movement: `rtl/blocks/tile_reader.sv`, `rtl/blocks/tile_writer.sv`, `rtl/blocks/pw_tile_reader.sv`
- layer runners: `rtl/blocks/conv1_tile_runner.sv`, `rtl/blocks/dws_tile_runner.sv`, `rtl/blocks/gap_runner.sv`, `rtl/blocks/fc_runner.sv`
- orchestration: `rtl/blocks/tile_ctrl.sv`, `rtl/blocks/mobilenet_v1_ctrl.sv`, `rtl/blocks/mobilenet_v1_top.sv`
- integration: `rtl/blocks/mobilenet_v1_bram_wrapper.sv`, `rtl/blocks/mobilenet_v1_axi_ctrl_bram.sv`, `rtl/blocks/mobilenet_v1_vivado_bram_top.sv`

## Important architectural decisions
## File-backed artifacts are the integration boundary
This codebase prefers file handoff over in-memory pipelines.
Examples:
- model parameters become `.mem` files in `rtl/mem/`
- reference outputs become `*_expected.mem`
- hardware results become `*_hw.mem`
- ROI logic produces masks in `.png`, `.npy`, and `.mem` formats

That makes debug and replay easy, but it also means many flows depend on directory conventions staying stable.

## Parameterization is concentrated in RTL module parameters
The RTL does not centralize configuration in one package.
`rtl/mobilenet_v1_pkg.sv` is minimal; real tuning knobs live as parameters on top modules such as `rtl/blocks/mobilenet_v1_top.sv` and `rtl/blocks/mobilenet_v1_axi_ctrl_bram.sv`.
Examples include `OC_PAR`, `PW_GROUP`, `PW_OC_PAR`, `PW_IC_PAR`, `TILE_H`, `TILE_W`, and `USE_XPM_RAM`.

## The control plane is layered
There is a clear wrapper ladder:
`mobilenet_v1_top` -> `mobilenet_v1_bram_wrapper` -> `mobilenet_v1_reg_shell` or AXI wrappers -> board-specific top modules.
That separation is useful for planning because register-map changes should usually happen above the raw accelerator core.

## Tile skipping is a first-class concern
ROI-aware execution is not an afterthought.
It appears in:
- mask-generation scripts like `scripts/gen_tile_mask_mem.py`
- control logic in `rtl/blocks/mobilenet_v1_ctrl.sv`
- register helpers in `sw/mobilenet_v1_ctrl_bram_regs.h`
- board bring-up examples that explicitly load masks before `start`

## Main Entry Points For Future Work
- Python model entry: `src/MobileNet_tf.py`
- NumPy golden entry: `src/MobileNet_np.py`
- weight export entry: `scripts/export_mobilenet_int8_mem.py`
- golden-output generation entry: `scripts/gen_golden_fc.py`
- accelerator core entry: `rtl/blocks/mobilenet_v1_top.sv`
- preferred SoC wrapper entry: `rtl/blocks/mobilenet_v1_axi_ctrl_bram.sv`
- Vivado packaging entry: `rtl/blocks/mobilenet_v1_vivado_bram_top.sv`
- PS bring-up entry: `sw/mobilenet_v1_vitis_bringup_example.c`

## Planning Notes
- Changes to layer order or channel counts must stay aligned across `src/MobileNet_tf.py`, `src/MobileNet_np.py`, exporter scripts, and `rtl/blocks/mobilenet_v1_ctrl.sv`.
- Many scripts assume repo-root execution and fixed output directories; workflow refactors should preserve or explicitly replace those assumptions.
- The repo contains both source and large generated assets. Planning work should distinguish between authored code in `src/`, `scripts/`, `rtl/`, `sw/` and disposable/generated state such as `obj_dir/`, `rtl/mem/`, `processed_frames/`, and dataset folders.
