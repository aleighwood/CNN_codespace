# External Integrations Map

## Overview
- This repo is not a web service; its integrations are dataset, ML framework, simulator, and FPGA-toolchain oriented.
- Most integrations are file-based and orchestrated by Python scripts under `scripts/`.
- The main cross-boundary handoffs are:
  Python model code -> TFLite artifacts -> RTL `.mem` files -> SystemVerilog testbenches -> Xilinx PS/PL software wrappers.

## TensorFlow / Keras Integration
- `src/MobileNet_tf.py` builds the reference MobileNet v1 model using TensorFlow Keras.
- `scripts/add_weights.py` integrates with `tf.keras.applications.MobileNet(weights="imagenet", ...)` to fetch pretrained ImageNet weights and save them to `mobilenet_imagenet.weights.h5`.
- `scripts/eval_imagenet_val.py` uses the same Keras model for floating-point evaluation on ImageNet-format validation images.
- `scripts/compare_models.py` compares the TensorFlow model against the NumPy implementation in `src/MobileNet_np.py`.
- This is the core software reference model used by the rest of the toolchain.

## TFLite Conversion And Inference Integration
- `scripts/quantize_from_imagenet_val.py` integrates with `tf.lite.TFLiteConverter` to produce int8 models such as `quantized_models/mobilenet_int8_ilsvrc2012.tflite`.
- `scripts/quantization_benchmark.py` emits multiple TFLite variants into `quantized_models/` and `quantized_models/calib_test*/`.
- `scripts/eval_imagenet_val.py`, `scripts/eval_roi_tile_skip.py`, `scripts/eval_golden_int8.py`, `scripts/compare_int8_paths.py`, `scripts/image_to_input_mem.py`, `scripts/image_to_input_mem_roi.py`, `scripts/gen_golden_fc.py`, and `scripts/export_mobilenet_int8_mem.py` all integrate with `tf.lite.Interpreter`.
- The TFLite model is treated as the source of truth for quantization parameters and, optionally, for exported weights when `--use-tflite-weights` is passed to `scripts/export_mobilenet_int8_mem.py`.
- Practical example: `scripts/export_mobilenet_int8_mem.py` reads `quantized_models/mobilenet_int8_ilsvrc2012_5000.tflite` and writes ROM-init files into `rtl/mem/`.

## TensorFlow Datasets / ImageNet Integration
- `scripts/build_imagenet_npz.py` integrates with `tensorflow_datasets` using `tfds.builder(...)` and `builder.download_and_prepare(...)`.
- That script supports both TFDS-managed datasets and manual ImageNet downloads via `--manual-dir`.
- Default dataset naming comes from TFDS (`imagenet2012` or `imagenet_v2/matched-frequency`), not from local custom loaders.
- Several evaluation and quantization scripts also integrate directly with a local ImageNet-style directory tree at `ILSVRC2012_val/`.
- Ground truth and metadata are read from `ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt` and `ILSVRC2012_devkit_t12/data/meta.mat` by scripts such as `scripts/eval_imagenet_val.py` and `scripts/eval_golden_int8.py`.

## Image Processing Integration
- Pillow is used across scripts to load, crop, draw, and save images, for example in `scripts/run_roi_golden.py`, `scripts/roi_visualize.py`, `scripts/roi_tile_overlay.py`, and `scripts/image_to_input_mem.py`.
- OpenCV (`cv2`) is used in `scripts/object_crop.py` for video capture, image reading/writing, resizing, bounding boxes, and mask output.
- `matplotlib` is used in `scripts/image_crop.py` for tile/window visualizations.
- These integrations are offline tooling only; there is no service or streaming interface.

## Transparent Background Integration
- `scripts/object_crop.py` integrates with `transparent_background.Remover` to generate masks and cropped foreground assets.
- The repo contains `.transparent-background/config.yaml`, which points to checkpoint URLs hosted on GitHub releases.
- Cached model artifacts already exist in `.transparent-background/ckpt_base.pth`.
- The default runtime in `scripts/object_crop.py` uses `device="cuda"`, so this path expects GPU-capable execution when available.
- Outputs from this integration are written into `selected_frames/`, `processed_frames/`, and optionally reused by ROI scripts such as `scripts/gen_tile_mask_mem.py`.

## Python Subprocess Orchestration
- Several utilities are composed by spawning other repo scripts instead of importing them as libraries.
- `scripts/run_roi_golden.py` shells out to image conversion and golden-generation commands with `subprocess.run(..., check=True)`.
- `scripts/eval_golden_int8.py` shells out to preprocessing and golden-generation scripts for each evaluation sample.
- `scripts/eval_roi_tile_skip.py`, `scripts/compare_int8_paths.py`, and `scripts/sweep_bitmap.py` also depend on subprocess chaining.
- This means integration boundaries inside the repo are CLI contracts plus filesystem outputs, not stable Python APIs.

## RTL Memory Export Integration
- `scripts/export_mobilenet_int8_mem.py` is the main bridge from ML artifacts to hardware-ready data.
- It reads Keras or TFLite weights and emits hex memories such as `rtl/mem/conv1_weight.mem`, `rtl/mem/dw_weight.mem`, `rtl/mem/pw_weight.mem`, and `rtl/mem/fc_weight.mem`.
- It also emits quantization parameter memories like `rtl/mem/conv1_mul.mem`, `rtl/mem/dw_shift.mem`, `rtl/mem/pw_bias_rq.mem`, and `rtl/mem/fc_zp.mem`.
- `rtl/blocks/mobilenet_v1_param_cache.sv` and `rtl/blocks/mobilenet_v1_top.sv` consume those files through `INIT_*` parameters and `$readmemh`.
- This is the repo’s most important cross-language integration point.

## RTL Simulation / Verification Integration
- Testbenches in `rtl/tb/` integrate with generated `.mem` data from the Python scripts.
- `rtl/tb/tb_mobilenet_v1_top.sv` expects inputs like `rtl/mem/input_rand.mem`, `rtl/mem/tile_mask.mem`, and the full set of exported parameter memories.
- The same testbench writes back hardware dumps such as `rtl/mem/fc_out_hw.mem`, `rtl/mem/fc_logits_hw.mem`, and per-layer outputs for later comparison.
- Checked-in build files in `obj_dir/` such as `obj_dir/Vtb_mobilenet_v1_top.mk` strongly suggest Verilator is the active simulator integration.
- There is no single wrapper script, so the simulator invocation details are external to the repo or run ad hoc by developers.

## Xilinx FPGA Toolchain Integration
- `rtl/blocks/mobilenet_v1_axi_lite.sv` exposes an AXI4-Lite slave for control/status integration.
- `rtl/blocks/mobilenet_v1_axi_ctrl_bram.sv` separates AXI-Lite control from a BRAM-style feature-map path.
- `rtl/blocks/mobilenet_v1_vivado_bram_top.sv` is tailored for Vivado block-design wiring with `bram_*_a` signals.
- `rtl/blocks/mobilenet_v1_htg_vsl5_top.sv` is the board-facing top intended for HTG-VSL5 bring-up.
- `rtl/common/dual_port_ram_async.sv` optionally instantiates `xpm_memory_tdpram` when `XILINX_XPM` is defined.
- `rtl/README.md` and `sw/README.md` both describe Vivado/Vitis as the intended downstream toolchain, even though no Vivado project files are checked in.

## Vitis / PS Software Integration
- `sw/mobilenet_v1_ctrl_bram_xil.h` integrates with Vitis conventions through `Xil_In32`, `Xil_Out32`, `UINTPTR`, and optional inclusion of `xparameters.h`.
- `sw/mobilenet_v1_vitis_bringup_example.c` assumes the control block and feature-map BRAM are memory-mapped into the PS address space.
- Fallback address macros like `XPAR_MOBILENET_CTRL_BASEADDR` and `XPAR_MOBILENET_FM_BRAM_BASEADDR` show the intended integration with generated BSP constants.
- `sw/mobilenet_v1_bringup_example.c` and `sw/mobilenet_v1_ctrl_bram_regs.h` expose the register-level software contract used to start inference, load tile masks, and poll completion.
- This side of the system is a bring-up skeleton, not a production firmware stack.

## Filesystem-Based Data Contracts
- The repo relies heavily on stable relative paths instead of service APIs.
- Input images, quantized models, and dataset directories are passed via CLI but default to paths like `ILSVRC2012_val/`, `quantized_models/...`, and `rtl/mem/...`.
- ROI scripts exchange masks through files such as `processed_frames/dog/dog_mask.npy` and `rtl/mem/roi_mask.npy`.
- Hardware/software comparisons are done by reading and writing `.mem` files rather than through sockets, RPC, or shared libraries.
- Any future refactor needs to preserve these file contracts or replace them systematically.

## Integration Risks And Planning Notes
- `requirements.txt` does not declare Pillow even though many scripts require `PIL`, so environment setup can fail silently on a clean machine.
- The Transparent Background integration persists model checkpoints inside the repo workspace, which mixes cache state with project state.
- No checked-in build automation ties together TensorFlow export, TFLite generation, RTL simulation, and Vitis bring-up.
- Toolchain assumptions are embedded in docs and code comments rather than enforced through CI or pinned project files.
- The strongest existing integration seams for future planning are:
  `src/MobileNet_tf.py` -> `quantized_models/*.tflite` -> `scripts/export_mobilenet_int8_mem.py` -> `rtl/mem/*.mem` -> `rtl/tb/*.sv` / `sw/*.c`.
