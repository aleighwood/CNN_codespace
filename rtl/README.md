RTL building blocks for MobileNet v1 (int8)

Overview
- `rtl/common/line_buffer_3x3.sv`: Stream to 3x3 window; expects padding already applied; stride gating supported.
- `line_buffer_3x3` takes runtime tile sizes (`cfg_img_w/h`) and stride (`cfg_stride`), and requires `MAX_IMG_W/H` large enough for the largest tile-in size.
- `line_buffer_3x3` supports a `start` pulse to reset its internal counters between tiles.
- `rtl/common/dw_conv_3x3.sv`: 9x MAC for a single channel window.
- `rtl/common/conv3x3_mac_vec.sv`: Vectorized 3x3 MAC for multiple output channels in parallel.
- `rtl/common/requant_q31.sv`: Q31 requantization (SRDHM + RoundingDivideByPOT) with optional ReLU6 clamp.
- `rtl/common/tile_buf.sv`: Simple sync RAM wrapper for tile storage.
- `rtl/common/tile_mask_mem.sv`: Synchronous tile-mask storage with 1-cycle read latency and a single write port for runtime mask updates.
- `rtl/common/dual_port_ram_async.sv`: Dual-port RAM wrapper used by the board-facing shell; defaults to behavioral async-read RAM for Verilator and can optionally instantiate XPM memory when `XILINX_XPM` is defined and `USE_XPM` is set.
- `rtl/blocks/depthwise_stage.sv`: line buffer + depthwise conv + requant.
- `rtl/blocks/pw_conv_1x1.sv`: Serial 1x1 dot-product across input channels.
- `rtl/blocks/dws_block.sv`: Skeleton wrapper that exposes depthwise and pointwise stages; tile buffer sits between them.
- `rtl/blocks/tile_ctrl.sv`: On-the-fly tile iterator that emits input/output tile origins and sizes.
- `rtl/blocks/tile_reader.sv`: Streams a single-channel input tile from flat memory, injecting `cfg_pad_value` for padding.
- `rtl/blocks/tile_writer.sv`: Writes a single-channel output tile into flat memory.
- `rtl/blocks/pw_tile_reader.sv`: Reads a depthwise tile in channel-major order for pointwise accumulation and exposes input-channel index.
- `rtl/blocks/dws_tile_runner.sv`: Coordinates per-tile depthwise and pointwise phases with external memory interfaces.
- `rtl/blocks/conv1_tile_runner.sv`: Tile-based 3x3 conv runner with partial-sum buffering and configurable output-channel parallelism.
- `rtl/blocks/mobilenet_v1_ctrl.sv`: Sequences conv1 + 13 depthwise blocks, drives tile iteration, and ping-pongs feature-map buffers.
- `rtl/blocks/mobilenet_v1_top.sv`: Top-level wrapper that wires the controller, tile runners, and parameter cache.
- `rtl/blocks/mobilenet_v1_bram_wrapper.sv`: Board-facing integration wrapper with host-loadable feature-map RAM, runtime tile-mask writes, and the existing core kept intact. `USE_XPM_RAM` propagates the optional XPM-backed memory path into the local RAM banks.
- `rtl/blocks/mobilenet_v1_reg_shell.sv`: Bus-neutral control/register shell around the BRAM wrapper; intended to be the clean handoff point for an AXI-Lite slave in Vivado.
- `rtl/blocks/mobilenet_v1_axi_lite.sv`: Thin AXI4-Lite slave in front of the register shell; intended for Vivado IP packaging and PS register access.
- `rtl/blocks/mobilenet_v1_axi_ctrl_bram.sv`: AXI4-Lite control wrapper around the BRAM wrapper with a direct host feature-map port. This is the preferred PS/PL bring-up path when you want AXI-Lite for control only and a separate bulk image buffer path.
- `rtl/blocks/mobilenet_v1_htg_vsl5_top.sv`: Final board-facing synthesis top for Vivado packaging on the HTG-VSL5; wraps `mobilenet_v1_axi_ctrl_bram` and exposes only PL-facing clock/reset, AXI-Lite control, direct feature-map ports, and `irq`.
- `rtl/blocks/mobilenet_v1_vivado_bram_top.sv`: Vivado-facing packaging top with standard AXI-Lite clock/reset names and Xilinx-style BRAM slave signal names (`bram_*_a`) for the feature-map window.
- `rtl/blocks/mobilenet_v1_param_cache.sv`: On-chip parameter cache/ROM with pointwise group cache (`PW_GROUP`) and a single write port; supports `$readmemh` init.
- `export_mobilenet_int8_mem.py`: Exports int8 weights and per-channel quant params into `.mem` files (hex) for ROM init.
- `rtl/blocks/gap_runner.sv`: Global average pooling over HxW per channel.
- `rtl/blocks/fc_runner.sv`: Fully-connected (1x1) classifier using cached weights and quant params.
- `rtl/tb/tb_param_cache.sv`: Simple sanity-check testbench for ROM init and parameter addressing.
- `rtl/tb/tb_mobilenet_v1_axi_lite.sv`: Control-plane unit test for AXI-Lite read/write sequencing and register-map behavior.

Memory implementation notes
- The default memory path is still the behavioral async-read model so the current Verilator regressions remain valid.
- If you define `XILINX_XPM` in Vivado and set `USE_XPM_RAM=1`, the wrapper instantiates `xpm_memory_tdpram` for the feature-map and depthwise scratch banks.
- If you set `USE_XPM_RAM=1` in Verilator without `XILINX_XPM`, the RAM wrapper now uses a synchronous behavioral model so the 1-cycle memory-latency path can be regression-tested locally.
- The XPM path uses 1-cycle synchronous reads, which matches real BRAM/URAM behavior.
- The inference datapath is now aligned to that 1-cycle read latency: `gap_runner` and `fc_runner` were retimed, and the tile readers were already latency-tolerant.
- The host-side feature-map readback path is still a simple bring-up interface, so software-facing read timing should be rechecked in Vivado if you rely on it heavily.
- `mobilenet_v1_param_cache` now carries explicit `ram_style` synthesis hints:
  - large weight stores (`dw_weight_mem`, `pw_weight_mem`, `fc_weight_mem`, `pw_cache_mem`) are marked for URAM
  - smaller per-channel tables are marked for BRAM
- These are still plain RTL arrays, not explicit XPM memories, because the current parameter path relies on combinational reads. This is the safe intermediate step before any full synchronous-parameter-memory retime.
- Vivado must still be used to confirm the inferred URAM/BRAM mapping on the target device.

Tiling model (tile buffering)
- Use tiles (ex: 16x16) with a 1-pixel halo for 3x3 pads. The halo can be injected in software or by a tile reader that inserts zeros.
- Depthwise stage runs one input channel at a time, writing a depthwise output tile.
- Depthwise tile buffer layout is channel-major: `[ch][row][col]`.
- Pointwise stage reads the depthwise tile in channel-major order per pixel (all channels for one pixel), feeding `pw_conv_1x1` via `pw_tile_reader`.
- Input/output feature maps are assumed to be planar (channel-major) with base addresses per channel.

Quantization notes
- Typical flow: int8 inputs/weights → int32 accumulation → Q31 requant (`mul_q31` + right shift).
- ReLU6 clamp uses `relu6_min = zp_out` and `relu6_max = round(6 / scale_out) + zp_out` in output quant units.

MobileNet v1 mapping
- The layer ordering and depthwise block strides match `MobileNet_np.py` and `DEPTHWISE_BLOCK_SPECS`.
- Use `conv1_tile_runner` for the initial 3x3 stride-2 layer; it expects weights/bias/quant params per output-channel group.
- `conv1_tile_runner` processes one input channel at a time and accumulates into on-chip psum buffers sized to the tile area.
- `mobilenet_v1_ctrl` uses MobileNet v1 layer specs and outputs tile parameters for either `conv1_tile_runner` or `dws_tile_runner`.
- `mobilenet_v1_param_cache` expects `layer_idx`, `layer_in_c`, and `layer_out_c` to resolve per-layer weight/param offsets.
- The head is implemented as `gap_runner` (global average pool) followed by `fc_runner` (classifier).

Parameter cache write selects (`param_wr_sel`)
- 0: conv1 weights, 1: conv1 bias_acc, 2: conv1 mul, 3: conv1 bias_requant, 4: conv1 shift, 5: conv1 relu6_max
- 6: depthwise weights, 7: depthwise mul, 8: depthwise bias_acc, 9: depthwise shift, 10: depthwise relu6_max
- 11: pointwise weights, 12: pointwise bias_acc, 13: pointwise mul, 14: pointwise bias_requant, 15: pointwise shift, 16: pointwise relu6_max
- 17: gap mul, 18: gap bias, 19: gap shift
- 20: fc weights, 21: fc mul, 22: fc bias_acc, 23: fc shift
- 24: conv1 relu6_min, 25: depthwise relu6_min, 26: pointwise relu6_min, 27: fc zp
- `mobilenet_v1_top` exposes `param_wr_*` signals (5-bit `param_wr_sel`) to load the parameter cache before running.
- Pointwise weights are cached in groups (`PW_GROUP`) and `dws_tile_runner` waits for `pw_group_ready` before each group.
- Use the `INIT_*` parameters on `mobilenet_v1_param_cache` to pre-initialize ROMs for bitstream builds.
- `mobilenet_v1_top` passes through `INIT_*` parameters so you can set file paths at the top level.

Simple register shell (`mobilenet_v1_reg_shell`)
- The register shell exposes a minimal synchronous register interface:
  - `reg_wr_en`, `reg_wr_addr`, `reg_wr_data`
  - `reg_rd_en`, `reg_rd_addr`, `reg_rd_data`, `reg_rd_valid`
  - `irq`
- It keeps a small software-visible register map that can later sit behind AXI-Lite without changing the accelerator core.
- Register map:
  - `0x00` control: bit 0 = start pulse, bit 1 = `tile_skip_en`, bit 2 = `irq_enable`, bit 4 = clear done sticky
  - `0x04` status: bit 0 = busy, bit 1 = done sticky, bit 2 = done pulse, bit 3 = irq
  - `0x08` input height
  - `0x0c` input width
  - `0x10` feature-map host address
  - `0x14` feature-map host write data
  - `0x18` feature-map host read data
  - `0x1c` feature-map command: bit 0 = write one byte at current feature-map address
  - `0x20` tile-mask address
  - `0x24` tile-mask data: bit 0 = tile keep/skip value to write
  - `0x28` tile-mask command: bit 0 = write one mask entry
  - `0x30` parameter write select
  - `0x34` parameter write address
  - `0x38` parameter write data
  - `0x3c` parameter command: bit 0 = issue one parameter write

AXI-Lite wrapper (`mobilenet_v1_axi_lite`)
- Exposes a single-outstanding AXI4-Lite slave for control/status access.
- Internally translates AXI transactions into the same register map used by `mobilenet_v1_reg_shell`.
- Intended as the module to package as custom IP in Vivado.
- Assumes 32-bit AXI data and 8-bit local register addressing by default.
- Software should use full 32-bit register writes during bring-up.

AXI-Lite + BRAM wrapper (`mobilenet_v1_axi_ctrl_bram`)
- Exposes the same single-outstanding AXI4-Lite control path for control/status, tile-mask writes, and parameter writes.
- Removes the byte-at-a-time feature-map register path from AXI-Lite.
- Instead, exposes direct host feature-map ports:
  - `host_fm_en`
  - `host_fm_we`
  - `host_fm_addr`
  - `host_fm_din`
  - `host_fm_dout`
- This is intentionally BRAM-native in style: one shared address port, byte write-enable, write data, and read data.
- This is intended for PS/PL systems where:
  - AXI-Lite configures and starts the accelerator
  - a separate BRAM or memory-mapped datapath loads the input tensor and reads outputs
- Control register map kept in this wrapper:
  - `0x00` control: bit 0 = start pulse, bit 1 = `tile_skip_en`, bit 2 = `irq_enable`, bit 4 = clear done sticky
  - `0x04` status: bit 0 = busy, bit 1 = done sticky, bit 2 = done pulse, bit 3 = irq
  - `0x08` input height
  - `0x0c` input width
  - `0x20` tile-mask address
  - `0x24` tile-mask data: bit 0 = tile keep/skip value to write
  - `0x28` tile-mask command: bit 0 = write one mask entry
  - `0x30` parameter write select
  - `0x34` parameter write address
  - `0x38` parameter write data
  - `0x3c` parameter command: bit 0 = issue one parameter write

HTG-VSL5 synthesis top (`mobilenet_v1_htg_vsl5_top`)
- This is the clean top-level module to package as custom IP in Vivado for the HTG-VSL5 bring-up flow.
- It is a thin board-facing shell around `mobilenet_v1_axi_ctrl_bram`.
- Exposed interfaces:
  - `pl_clk`, `pl_rst_n`
  - a single AXI4-Lite slave port for control/status
  - a BRAM-native host feature-map port for bulk input/output
  - `irq`
- `USE_XPM_RAM` defaults to `1` here because this is intended for the synthesis path, but it remains parameterized so you can override it during bring-up if needed.

Vivado BRAM packaging top (`mobilenet_v1_vivado_bram_top`)
- This is the most convenient top to package if you want the feature-map buffer to look like a Xilinx native BRAM slave in block design wiring.
- It exposes:
  - `s_axi_aclk`, `s_axi_aresetn`
  - the AXI4-Lite control/status slave
  - `bram_rst_a`, `bram_clk_a`, `bram_en_a`, `bram_we_a`, `bram_addr_a`, `bram_wrdata_a`, `bram_rddata_a`
  - `irq`
- Internally it still uses one shared clock domain, so in Vivado the BRAM and AXI-Lite sides should be tied to the same PL clock/reset.

ROM init workflow (hex, one file per array)
- Run: `python export_mobilenet_int8_mem.py --weights mobilenet_imagenet.weights.h5 --output-dir rtl/mem`
- Pass these filenames into `mobilenet_v1_param_cache` via `INIT_*` parameters:
  - conv1: `conv1_weight.mem`, `conv1_bias_acc.mem`, `conv1_mul.mem`, `conv1_bias_rq.mem`, `conv1_shift.mem`, `conv1_relu6.mem`
  - conv1: `conv1_weight.mem`, `conv1_bias_acc.mem`, `conv1_mul.mem`, `conv1_bias_rq.mem`, `conv1_shift.mem`, `conv1_relu6.mem`, `conv1_relu6_min.mem`
  - depthwise: `dw_weight.mem`, `dw_mul.mem`, `dw_bias_acc.mem`, `dw_shift.mem`, `dw_relu6.mem`, `dw_relu6_min.mem`
  - pointwise: `pw_weight.mem`, `pw_bias_acc.mem`, `pw_mul.mem`, `pw_bias_rq.mem`, `pw_shift.mem`, `pw_relu6.mem`, `pw_relu6_min.mem`
  - gap: `gap_mul.mem`, `gap_bias.mem`, `gap_shift.mem`
  - fc: `fc_weight.mem`, `fc_mul.mem`, `fc_bias_acc.mem`, `fc_shift.mem`, `fc_zp.mem`

Quantization assumptions in exporter
- Per-channel symmetric int8 weights with per-tensor activation scales/zero-points from TFLite.
- ReLU6 outputs use `relu6_min = zp_out` and `relu6_max = round(6 / scale_out) + zp_out`.
- BN is fused into weights and bias before quantization.
- Bias is applied in the accumulator (`bias_acc`); requant uses `mul_q31` and right shifts.
