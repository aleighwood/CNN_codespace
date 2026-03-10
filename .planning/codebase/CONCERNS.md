# Codebase Concerns

## Overall Risk Profile

This repository is doing real cross-domain work: Python reference modeling, quantization/export tooling, RTL implementation, testbenches, and PS-side bring-up helpers. The main risk is not one obviously broken module; it is drift between these layers because architecture constants, memory contracts, and validation logic are spread across many files with only light automation binding them together.

## High-Risk Maintenance Hotspots

### 1. Architecture constants are duplicated across Python, RTL, and software

- The MobileNet topology is defined independently in `src/MobileNet_np.py`, `src/MobileNet_tf.py`, `rtl/mobilenet_v1_pkg.sv`, and deeply inside `rtl/blocks/mobilenet_v1_param_cache.sv`.
- Input/image assumptions such as `224x224x3`, tile sizes, and `1000` output classes also appear in `scripts/add_weights.py`, `scripts/eval_imagenet_val.py`, `scripts/compare_int8_paths.py`, `sw/mobilenet_v1_bringup_example.c`, and `sw/mobilenet_v1_vitis_bringup_example.c`.
- This means a model-shape change or even a minor quantization contract change has multiple manual update points. The likely failure mode is silent mismatch, not a clean compile error.

### 2. Verification is present, but much of it is manual and artifact-driven

- There is no `pyproject.toml`, `requirements.txt`, `environment.yml`, `pytest.ini`, or CI wiring in the repo root, so the supported validation flow is mostly tribal knowledge plus direct script usage.
- `rtl/tb/tb_mobilenet_v1_top.sv` is useful for observability, but it mostly dumps `.mem` artifacts and exits on `done` or timeout; it does not self-check against expected golden outputs inside the testbench.
- Several comparison utilities such as `scripts/compare_fc.py`, `scripts/compare_fc_logits.py`, `scripts/compare_all_layers.py`, and `scripts/compare_int8_paths.py` sit outside an automated regression harness.
- The result is that regressions can easily become "someone forgot to rerun the compare script" problems.

### 3. Environment reproducibility is weak

- `scripts/README.md` refers to a `mobilenet_env` conda environment, but that environment definition is not stored in the repository.
- Python scripts depend on heavy packages like TensorFlow, OpenCV, Pillow, and `transparent_background`, but there is no lockfile or install manifest that pins versions.
- Because quantization and preprocessing are numerically sensitive, unpinned dependency drift can change outputs without any source-level code changes.

## Python Model And Tooling Concerns

### 4. The "NumPy" path still depends on TensorFlow imports

- `src/conv.py`, `src/depthwise_conv.py`, and `src/Fully_connected.py` import TensorFlow even though their main inference functions are NumPy implementations.
- That makes the supposedly lightweight reference path harder to run in minimal environments and couples core math helpers to the heaviest dependency in the repo.

### 5. Core reference ops are straightforward but slow and lightly guarded

- `src/conv.py`, `src/depthwise_conv.py`, `src/pooling.py`, and `src/Fully_connected.py` use deeply nested Python loops rather than vectorized kernels.
- That is acceptable for educational code or small sanity tests, but it becomes a performance bottleneck and makes broad golden-generation sweeps expensive.
- Shape validation is thin. For example, `src/MobileNet_np.py` only explicitly checks pointwise filter count; it does not comprehensively validate batch-norm vector sizes, FC dimensions, or channel compatibility before compute starts.
- Several outputs are created with default NumPy dtypes (`np.zeros(...)` without explicit dtype), which can quietly promote calculations to `float64` and make parity debugging harder.

### 6. Script orchestration is CLI-to-CLI, not library-driven

- `scripts/run_roi_golden.py`, `scripts/compare_int8_paths.py`, `scripts/eval_golden_int8.py`, `scripts/eval_roi_tile_skip.py`, and `scripts/sweep_bitmap.py` call sibling scripts through `subprocess.run(...)`.
- This creates duplicated argument contracts and path assumptions instead of shared Python APIs.
- Small CLI changes in `scripts/image_to_input_mem.py` or `scripts/gen_golden_fc.py` can break multiple downstream flows.

### 7. Several scripts assume shared writable output locations

- Many scripts write directly into `rtl/mem`, `selected_frames`, or `processed_frames`.
- Examples include `scripts/run_roi_golden.py`, `scripts/compare_int8_paths.py`, `scripts/object_crop.py`, and `scripts/export_mobilenet_int8_mem.py`.
- That makes parallel runs unsafe and increases the chance of stale artifacts contaminating the next validation pass.

### 8. There is dead or scratch code mixed into the supported tool surface

- `scripts/not_in_use.py` is explicitly documented as old scratch code and is also visibly broken or incomplete.
- Leaving this file in the active scripts directory increases noise for future contributors and makes it harder to tell what is actually supported.

### 9. External assets are fetched or interpreted without much provenance control

- `scripts/add_weights.py` pulls ImageNet MobileNet weights but does not record checksum, source version, or expected framework version.
- Quantized model flows in `scripts/quantize_from_imagenet_val.py` and `scripts/export_mobilenet_int8_mem.py` assume the local `.tflite` artifact is the correct one, but there is no manifest tying generated `.mem` files back to a specific source model.
- This is more an operational integrity risk than an exploit risk, but it matters because stale or mismatched artifacts can look like RTL bugs.

### 10. The ImageNet metadata path contains bespoke parsing logic

- `scripts/eval_imagenet_val.py` includes a handwritten MAT-file parser for `meta.mat`.
- That parser is non-trivial, domain-specific, and hard to validate without reference fixtures.
- It increases maintenance cost compared with either a simpler dependency-backed reader or checked-in preprocessed metadata.

### 11. Some scripts assume GPU-backed preprocessing by default

- `scripts/object_crop.py` defaults to `device="cuda"` and writes multiple raw/processed artifacts per input.
- On machines without CUDA, the default path is more likely to fail than to work.
- On large videos or directories, the script can also consume disk space aggressively because there is no retention or cleanup policy.

## RTL And HW/SW Interface Concerns

### 12. The register contract is duplicated in too many places

- Register offsets and bit semantics are effectively repeated in `rtl/blocks/mobilenet_v1_reg_shell.sv`, `rtl/blocks/mobilenet_v1_axi_ctrl_bram.sv`, `rtl/README.md`, `sw/mobilenet_v1_ctrl_bram_regs.h`, `sw/mobilenet_v1_ctrl_bram_xil.h`, and the AXI testbenches.
- This is a classic drift risk. One register-map edit requires synchronized updates in RTL, docs, software headers, and tests.
- A generated single-source register definition would reduce this sharply.

### 13. `mobilenet_v1_param_cache` is a central fragility point

- `rtl/blocks/mobilenet_v1_param_cache.sv` owns layer-index interpretation, memory initialization, quant parameter storage, pointwise group caching, and several offset computations.
- It also carries many baked-in architectural assumptions such as layer ordering, cache grouping, and class counts.
- This is the kind of file where small edits can cause valid synthesis but incorrect inference. It deserves targeted tests and clearer decomposition before major feature work lands on top of it.

### 14. Simulation and hardware memory behaviors are intentionally different

- `rtl/common/dual_port_ram_async.sv` and the notes in `rtl/README.md` describe multiple memory modes: behavioral async-read, optional XPM-backed memory, and a synchronous fallback mode.
- That flexibility is useful, but it also means there is a built-in sim-versus-silicon risk area.
- The more the datapath relies on exact read latency or BRAM inference behavior, the more this should be covered by explicit regression cases rather than documentation alone.

### 15. Bring-up software is easy to mistake for a real application

- `sw/mobilenet_v1_bringup_example.c` and `sw/mobilenet_v1_vitis_bringup_example.c` are honest about being skeletons, but they still contain realistic-looking base addresses, tensor loading, and output probing.
- A future user can easily copy these into a board flow without replacing the placeholder memory map or output offsets.
- The biggest practical risk here is false confidence during hardware bring-up.

### 16. Busy-wait control flow is acceptable for demos but weak for production use

- `sw/mobilenet_v1_ctrl_bram_regs.h` uses polling loops in `mobilenet_ctrl_bram_wait_done(...)` with no backoff, timeout logging, or interrupt-first path.
- That is fine for early bring-up, but it is not a robust software contract for a system that might later handle repeated inference or mixed workloads.

### 17. ROI/tile-skip behavior spans multiple layers of the stack without one source of truth

- Tile-skip semantics appear in RTL (`rtl/blocks/mobilenet_v1_ctrl.sv`, `rtl/blocks/mobilenet_v1_axi_ctrl_bram.sv`, `rtl/blocks/mobilenet_v1_reg_shell.sv`), software headers (`sw/mobilenet_v1_ctrl_bram_regs.h`), and preprocessing/golden scripts such as `scripts/run_roi_golden.py`, `scripts/gen_tile_mask_mem.py`, and `scripts/eval_roi_tile_skip.py`.
- This feature crosses preprocessing, memory layout, register programming, and inference scheduling, but there is no compact spec file that defines the end-to-end contract.
- That makes the ROI path more fragile than the standard full-frame path.

## Practical Planning Priorities

1. Create one generated model/spec source for layer layout, image shape, class count, and register map; consume it from Python, RTL docs, and C headers.
2. Turn the current artifact-comparison scripts into automated regressions, especially for `rtl/tb/tb_mobilenet_v1_top.sv` and the quantized export path.
3. Add a checked-in environment definition so `scripts/README.md` describes a reproducible setup rather than an implicit local convention.
4. Separate supported scripts from scratch code, and move subprocess-based script chaining toward shared Python library functions.
5. Add focused tests around `rtl/blocks/mobilenet_v1_param_cache.sv` and the async/sync RAM behavior split, because those look like the most likely sources of subtle inference mismatches.
