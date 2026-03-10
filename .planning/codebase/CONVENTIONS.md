# Code Conventions Map

## Scope

This repository mixes three code styles:

- Python library code under `src/`
- Python CLI and analysis utilities under `scripts/`
- SystemVerilog RTL and benches under `rtl/`, plus C bring-up code under `sw/`

There is no single enforced formatter or linter config in the repo root. Future work should treat the current state as "documented local conventions" rather than "strictly enforced standards."

## Repository-Level Patterns

- Paths are domain-oriented rather than package-oriented: `src/` holds reference model code, `scripts/` holds operational tooling, `rtl/common/` and `rtl/blocks/` split reusable primitives from composed hardware blocks, and `rtl/tb/` contains benches.
- Naming is descriptive and hardware-centric. Files usually encode the unit or flow directly, for example `rtl/blocks/mobilenet_v1_top.sv`, `rtl/common/line_buffer_3x3.sv`, and `scripts/export_mobilenet_int8_mem.py`.
- Generated or simulator-produced artifacts appear to be checked into the tree in `obj_dir/`. Planning work should avoid treating that directory as source-of-truth.

## Python Naming And Style

- Most `scripts/*.py` files use `snake_case` filenames and function names, for example `scripts/image_to_input_mem.py`, `scripts/compare_all_layers.py`, and `scripts/run_roi_golden.py`.
- `src/` is less consistent and mixes capitalized module filenames with snake_case modules: `src/Fully_connected.py`, `src/ReLU6.py`, `src/MobileNet_np.py`, `src/conv.py`, `src/pooling.py`.
- Class names follow standard Python casing where present. `src/MobileNet_np.py` defines `MobileNetV1Numpy`.
- Constants are uppercase when treated as configuration tables, for example `DEPTHWISE_BLOCK_SPECS` in `src/MobileNet_np.py`.
- Helper functions are usually short and procedural: `parse_shape`, `to_hex8`, `read_mem8`, `bbox_from_mask`, `compare_files`.
- Most CLI scripts follow the same entrypoint pattern:
  - build an `argparse.ArgumentParser`
  - parse into `args`
  - perform file/model work in `main()`
  - exit via `raise SystemExit(main())`
- Imports are mostly standard-library first, then third-party, then local imports. `scripts/eval_golden_int8.py` and `scripts/mobilenet_test.py` insert `REPO_ROOT` into `sys.path` to make local imports work when executed as scripts.

## Python Implementation Style

- The codebase favors explicit loops and intermediate variables over compact vectorized abstractions. This is especially visible in `src/conv.py` and `src/Fully_connected.py`.
- Docstrings are common in `src/` modules, but much thinner in `scripts/`. The core reference-model code is better documented than the operational utilities.
- Type hints are mostly absent. Argument and shape expectations are documented in docstrings or enforced at runtime instead.
- The scripts prefer direct file IO with `open(..., encoding="utf-8")` and `Path(...).parent.mkdir(parents=True, exist_ok=True)` when outputs are created, as in `scripts/image_to_input_mem.py`.
- Logging infrastructure is not used. Status and diagnostics are printed directly with `print(...)`.

## Error Handling Conventions

- CLI scripts generally fail fast with `SystemExit` and a specific message when input files, shapes, or quantization assumptions are wrong.
- Common examples:
  - `scripts/image_to_input_mem.py` rejects unsupported normalization modes.
  - `scripts/image_to_input_mem_roi.py` rejects missing images and empty ROIs.
  - `scripts/export_mobilenet_int8_mem.py` aborts on unexpected TFLite quantization metadata.
  - `scripts/eval_golden_int8.py` aborts before full evaluation if the sanity check fails.
- Reusable library code is more likely to raise `ValueError` than `SystemExit`. `src/MobileNet_np.py` uses `ValueError` for parameter-shape mismatches.
- Subprocess orchestration usually relies on `subprocess.run(..., check=True)` instead of manual return-code inspection, for example in `scripts/run_roi_golden.py`, `scripts/eval_golden_int8.py`, and `scripts/sweep_bitmap.py`.
- Error messages are usually concrete and actionable because they echo the invalid path, shape, or quantization state.

## RTL Conventions

- RTL module filenames and module names use lower snake case, for example `rtl/common/line_buffer_3x3.sv`, `rtl/common/requant_q31.sv`, and `rtl/blocks/mobilenet_v1_axi_lite.sv`.
- Testbenches use the `tb_` prefix: `rtl/tb/tb_line_buffer_3x3.sv`, `rtl/tb/tb_mobilenet_v1_axi_lite.sv`, `rtl/tb/tb_mobilenet_v1_top.sv`.
- Parameter blocks are heavily used and typed with `parameter int`, which makes width/config assumptions explicit.
- Port declarations use `logic` consistently, including signed packed arrays where needed.
- Internal constants are usually `localparam int` or `localparam logic [...]`.
- Sequential logic uses `always_ff @(posedge clk or negedge rst_n)` and active-low reset naming (`rst_n`) is consistent across sampled files.
- Handshake signals use standard names like `valid`, `ready`, `start`, `done`, and `busy`.
- Hierarchical organization is meaningful:
  - `rtl/common/` holds reusable primitives
  - `rtl/blocks/` holds composed datapath and shell blocks
  - `rtl/mobilenet_v1_pkg.sv` centralizes package-level definitions

## Verification-Oriented RTL Style

- Benches use simple self-check helpers instead of SVA or UVM. `rtl/tb/tb_mobilenet_v1_axi_lite.sv` and related benches implement `check_equal(...)`, accumulate `failures`, and print `TB PASSED` or `TB FAILED`.
- Debug visibility is intentionally built into top-level benches. `rtl/tb/tb_mobilenet_v1_top.sv` declares many "last_*" state trackers and debug counters to support long-running diagnosis.
- `$display` is preferred over richer reporting infrastructure.

## C Bring-Up Style

- `sw/mobilenet_v1_bringup_example.c` and related headers follow a straightforward embedded style: uppercase `#define` constants, `static` helpers, explicit pointer casts to memory-mapped regions, and polling-based completion.
- Comments are practical and hardware-integration focused, especially around address replacement and demo data assumptions.

## Consistency Gaps To Plan Around

- Python naming is inconsistent between `src/` and `scripts/`, especially `src/Fully_connected.py` and `src/ReLU6.py`.
- Formatting is not uniformly applied. Some files use tight spacing and older educational style (`src/conv.py`), while newer scripts are much closer to modern PEP 8.
- There is no visible linting, static typing, or formatter configuration.
- Runtime validation is strong in scripts, but code reuse boundaries are loose because many workflows communicate via generated `.mem` files and subprocess chaining rather than shared abstractions.

## Practical Guidance For Future Changes

- Match the local style of the area you touch rather than trying to normalize the whole repo opportunistically.
- For new CLI utilities under `scripts/`, follow the established `argparse` + `main()` + `SystemExit` pattern.
- For reusable Python model code under `src/`, prefer clearer shape checks and docstrings over cleverness.
- For RTL, preserve existing signal vocabulary (`*_valid`, `*_ready`, `cfg_*`, `*_wr_*`, `*_rd_*`) and parameterized widths.
- If introducing repo-wide quality tooling, do it as an explicit cleanup phase; current code does not appear ready for silent formatter or naming normalization.
