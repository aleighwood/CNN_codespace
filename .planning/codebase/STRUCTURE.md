# Repository Structure

## Top-Level Layout
The repository root mixes source code, generated artifacts, and local datasets.
That matters for planning because not every top-level directory should be treated as source.

```text
/home/simon/code/Project/CNN_codespace
|- src/
|- scripts/
|- rtl/
|- sw/
|- quantized_models/
|- ILSVRC2012_val/
|- ILSVRC2012_devkit_t12/
|- processed_frames/
|- selected_frames/
|- obj_dir/
|- .planning/
```

## Source Directories
## `src/`
Purpose: reference Python implementation of MobileNet v1 and its primitive ops.

Important files:
- `src/MobileNet_tf.py`: Keras/TF model definition plus `export_weights_to_numpy_dict()`
- `src/MobileNet_np.py`: NumPy inference model using explicit layer math
- `src/conv.py`: standard convolution helpers
- `src/depthwise_conv.py`: depthwise and depthwise-separable helpers
- `src/Fully_connected.py`: classifier math
- `src/batchnorm.py`, `src/pooling.py`, `src/ReLU6.py`, `src/softmax.py`, `src/zero_pad.py`: single-purpose operators

Structural note:
`src/__init__.py` is effectively empty, so this folder behaves as a light import namespace rather than a packaged library.

## `scripts/`
Purpose: command-line utilities for data prep, quantization, export, validation, ROI analysis, and visualization.

Representative clusters:
- model/export: `scripts/add_weights.py`, `scripts/quantize_from_imagenet_val.py`, `scripts/export_mobilenet_int8_mem.py`
- evaluation: `scripts/eval_imagenet_val.py`, `scripts/eval_golden_int8.py`, `scripts/compare_models.py`
- hardware comparison: `scripts/compare_all_layers.py`, `scripts/compare_fc.py`, `scripts/compare_fc_logits.py`
- ROI/tile skip: `scripts/eval_roi_tile_skip.py`, `scripts/run_roi_golden.py`, `scripts/sweep_bitmap.py`, `scripts/gen_tile_mask_mem.py`
- preprocessing/visualization: `scripts/object_crop.py`, `scripts/image_to_input_mem.py`, `scripts/roi_preview.py`, `scripts/roi_tile_overlay.py`

Structural note:
This folder is flat. There are no subpackages for shared utilities, so code reuse happens by copying small helpers or importing from `src/`.

## `rtl/`
Purpose: synthesizable SystemVerilog, simulation testbenches, and generated memory assets.

Subdirectories:
- `rtl/common/`: reusable arithmetic, memory, and buffering primitives
- `rtl/blocks/`: pipeline blocks, top modules, wrappers, and control logic
- `rtl/tb/`: testbenches, named consistently with `tb_*`
- `rtl/mem/`: generated `.mem` files plus some debug images and mask artifacts

Representative files:
- `rtl/blocks/mobilenet_v1_top.sv`: accelerator core
- `rtl/blocks/mobilenet_v1_ctrl.sv`: per-layer/tile orchestration
- `rtl/blocks/mobilenet_v1_axi_ctrl_bram.sv`: preferred PS/PL integration wrapper
- `rtl/blocks/mobilenet_v1_vivado_bram_top.sv`: Vivado packaging surface
- `rtl/mobilenet_v1_pkg.sv`: minimal shared typedef/package file

## `sw/`
Purpose: host-side register access helpers and Vitis bring-up examples.

Important files:
- `sw/mobilenet_v1_ctrl_bram_regs.h`: generic C register helper
- `sw/mobilenet_v1_ctrl_bram_xil.h`: Xilinx-specific helper using `Xil_In32` and `Xil_Out32`
- `sw/mobilenet_v1_vitis_bringup_example.c`: preferred BRAM-backed example
- `sw/mobilenet_v1_bringup_example.c`: older AXI-Lite-only feature-map path

## Generated and Environment-Specific Directories
## `rtl/mem/`
This is the main working artifact directory for simulation.
It contains:
- parameter memories such as `rtl/mem/conv1_weight.mem`, `rtl/mem/pw_weight.mem`, `rtl/mem/fc_weight.mem`
- input memories such as `rtl/mem/input_rand.mem` and `rtl/mem/input_dog.mem`
- expected outputs such as `rtl/mem/layer0_out_exp.mem` and `rtl/mem/fc_expected.mem`
- hardware captures such as `rtl/mem/layer0_out_hw.mem` and `rtl/mem/fc_out_hw.mem`
- ROI debug products such as `rtl/mem/roi_mask.npy` and overlay PNGs

## `quantized_models/`
Stores generated `.tflite` models and calibration outputs.
Scripts such as `scripts/quantize_from_imagenet_val.py` and `scripts/eval_imagenet_val.py` point here by default.

## `processed_frames/` and `selected_frames/`
These are outputs of `scripts/object_crop.py`.
The naming convention groups assets by input stem, for example `processed_frames/dog/` and `selected_frames/dog/`.

## `ILSVRC2012_val/` and `ILSVRC2012_devkit_t12/`
Local ImageNet validation assets used by evaluation and quantization scripts.
These are data dependencies, not source directories.

## `obj_dir/`
Verilator-style build output. Treat it as disposable generated state.

## Hidden Project Metadata
## `.planning/`
Planning workspace used by GSD/code-mapping flows.
This task owns `.planning/codebase/ARCHITECTURE.md` and `.planning/codebase/STRUCTURE.md`.

## `.settings/`, `.cache/`, `.transparent-background/`
Local tooling state. These support execution but are not part of the core code architecture.

## Naming Patterns
## Python
- Model files use mixed historical naming like `src/MobileNet_tf.py`, `src/MobileNet_np.py`, and `src/Fully_connected.py`
- Utility modules are short lowercase names such as `src/conv.py` and `src/pooling.py`
- Script names are verb-first and task-specific: `export_*`, `eval_*`, `compare_*`, `gen_*`, `image_*`, `roi_*`

## RTL
- Top-level accelerator and wrappers use the `mobilenet_v1_*` prefix
- execution units often end in `_runner`, for example `rtl/blocks/conv1_tile_runner.sv` and `rtl/blocks/fc_runner.sv`
- memory/data movers use `_reader`, `_writer`, `_buf`, or `_cache`
- testbenches consistently use the `tb_` prefix in `rtl/tb/`

## Software headers
- Generic helper headers end in `_regs.h`
- Xilinx-specific headers add `_xil.h`
- example programs end in `_example.c`

## Practical Reading Order
For future onboarding, the fastest order is:
1. `scripts/README.md`
2. `rtl/README.md`
3. `src/MobileNet_tf.py`
4. `src/MobileNet_np.py`
5. `scripts/export_mobilenet_int8_mem.py`
6. `rtl/blocks/mobilenet_v1_top.sv`
7. `rtl/blocks/mobilenet_v1_axi_ctrl_bram.sv`
8. `sw/mobilenet_v1_vitis_bringup_example.c`

## Structure Risks For Planning
- Source and generated artifacts live close together, so cleanup-sensitive work should avoid broad file globs.
- The flat `scripts/` layout makes duplication easy; extracting shared helpers would require care because many scripts are standalone CLIs.
- Some top-level files such as `requirements.txt`, `environment.yml`, weights, PDFs, and datasets are operational dependencies rather than architecture-defining code.
