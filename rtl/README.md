RTL building blocks for MobileNet v1 (int8)

Overview
- `rtl/common/line_buffer_3x3.sv`: Stream to 3x3 window; expects padding already applied; stride gating supported.
- `line_buffer_3x3` takes runtime tile sizes (`cfg_img_w/h`) and stride (`cfg_stride`), and requires `MAX_IMG_W/H` large enough for the largest tile-in size.
- `line_buffer_3x3` supports a `start` pulse to reset its internal counters between tiles.
- `rtl/common/dw_conv_3x3.sv`: 9x MAC for a single channel window.
- `rtl/common/conv3x3_mac_vec.sv`: Vectorized 3x3 MAC for multiple output channels in parallel.
- `rtl/common/requant_relu6.sv`: Per-layer scale and bias with optional ReLU6 clamp.
- `rtl/common/tile_buf.sv`: Simple sync RAM wrapper for tile storage.
- `rtl/blocks/depthwise_stage.sv`: line buffer + depthwise conv + requant.
- `rtl/blocks/pw_conv_1x1.sv`: Serial 1x1 dot-product across input channels.
- `rtl/blocks/dws_block.sv`: Skeleton wrapper that exposes depthwise and pointwise stages; tile buffer sits between them.
- `rtl/blocks/tile_ctrl.sv`: On-the-fly tile iterator that emits input/output tile origins and sizes.
- `rtl/blocks/tile_reader.sv`: Streams a single-channel input tile from flat memory, injecting zeros for padding.
- `rtl/blocks/tile_writer.sv`: Writes a single-channel output tile into flat memory.
- `rtl/blocks/pw_tile_reader.sv`: Reads a depthwise tile in channel-major order for pointwise accumulation and exposes input-channel index.
- `rtl/blocks/dws_tile_runner.sv`: Coordinates per-tile depthwise and pointwise phases with external memory interfaces.
- `rtl/blocks/conv1_tile_runner.sv`: Tile-based 3x3 conv runner with partial-sum buffering and configurable output-channel parallelism.
- `rtl/blocks/mobilenet_v1_ctrl.sv`: Sequences conv1 + 13 depthwise blocks, drives tile iteration, and ping-pongs feature-map buffers.
- `rtl/blocks/mobilenet_v1_top.sv`: Top-level wrapper that wires the controller, tile runners, and parameter cache.
- `rtl/blocks/mobilenet_v1_param_cache.sv`: On-chip parameter cache/ROM with pointwise group cache (`PW_GROUP`) and a single write port; supports `$readmemh` init.
- `export_mobilenet_int8_mem.py`: Exports int8 weights and per-channel quant params into `.mem` files (hex) for ROM init.
- `rtl/blocks/gap_runner.sv`: Global average pooling over HxW per channel.
- `rtl/blocks/fc_runner.sv`: Fully-connected (1x1) classifier using cached weights and quant params.
- `rtl/tb/tb_param_cache.sv`: Simple sanity-check testbench for ROM init and parameter addressing.

Tiling model (tile buffering)
- Use tiles (ex: 16x16) with a 1-pixel halo for 3x3 pads. The halo can be injected in software or by a tile reader that inserts zeros.
- Depthwise stage runs one input channel at a time, writing a depthwise output tile.
- Depthwise tile buffer layout is channel-major: `[ch][row][col]`.
- Pointwise stage reads the depthwise tile in channel-major order per pixel (all channels for one pixel), feeding `pw_conv_1x1` via `pw_tile_reader`.
- Input/output feature maps are assumed to be planar (channel-major) with base addresses per channel.

Quantization notes
- Typical flow: int8 inputs and weights, int32 accumulation, requant with per-layer `mul` and `shift`.
- ReLU6 clamp uses `relu6_max = floor(6 / scale_out)` in output quant units.

MobileNet v1 mapping
- The layer ordering and depthwise block strides match `MobileNet_np.py` and `DEPTHWISE_BLOCK_SPECS`.
- Use `conv1_tile_runner` for the initial 3x3 stride-2 layer; it expects weights/bias/quant params per output-channel group.
- `conv1_tile_runner` processes one input channel at a time and accumulates into on-chip psum buffers sized to the tile area.
- `mobilenet_v1_ctrl` uses MobileNet v1 layer specs and outputs tile parameters for either `conv1_tile_runner` or `dws_tile_runner`.
- `mobilenet_v1_param_cache` expects `layer_idx`, `layer_in_c`, and `layer_out_c` to resolve per-layer weight/param offsets.
- The head is implemented as `gap_runner` (global average pool) followed by `fc_runner` (classifier).

Parameter cache write selects (`param_wr_sel`)
- 0: conv1 weights, 1: conv1 bias_acc, 2: conv1 mul, 3: conv1 bias_requant, 4: conv1 shift, 5: conv1 relu6_max
- 6: depthwise weights, 7: depthwise mul, 8: depthwise bias, 9: depthwise shift, 10: depthwise relu6_max
- 11: pointwise weights, 12: pointwise bias_acc, 13: pointwise mul, 14: pointwise bias_requant, 15: pointwise shift, 16: pointwise relu6_max
- 17: gap mul, 18: gap bias, 19: gap shift
- 20: fc weights, 21: fc mul, 22: fc bias, 23: fc shift
- `mobilenet_v1_top` exposes `param_wr_*` signals (5-bit `param_wr_sel`) to load the parameter cache before running.
- Pointwise weights are cached in groups (`PW_GROUP`) and `dws_tile_runner` waits for `pw_group_ready` before each group.
- Use the `INIT_*` parameters on `mobilenet_v1_param_cache` to pre-initialize ROMs for bitstream builds.
- `mobilenet_v1_top` passes through `INIT_*` parameters so you can set file paths at the top level.

ROM init workflow (hex, one file per array)
- Run: `python export_mobilenet_int8_mem.py --weights mobilenet_imagenet.weights.h5 --output-dir rtl/mem`
- Pass these filenames into `mobilenet_v1_param_cache` via `INIT_*` parameters:
  - conv1: `conv1_weight.mem`, `conv1_bias_acc.mem`, `conv1_mul.mem`, `conv1_bias_rq.mem`, `conv1_shift.mem`, `conv1_relu6.mem`
  - depthwise: `dw_weight.mem`, `dw_mul.mem`, `dw_bias.mem`, `dw_shift.mem`, `dw_relu6.mem`
  - pointwise: `pw_weight.mem`, `pw_bias_acc.mem`, `pw_mul.mem`, `pw_bias_rq.mem`, `pw_shift.mem`, `pw_relu6.mem`
  - gap: `gap_mul.mem`, `gap_bias.mem`, `gap_shift.mem`
  - fc: `fc_weight.mem`, `fc_mul.mem`, `fc_bias.mem`, `fc_shift.mem`

Quantization assumptions in exporter
- Symmetric int8 weights/activations (zero-point = 0).
- ReLU6 outputs use `scale = 6/127` by default; input scale defaults to `1/127` (assumes input normalized to [-1,1]).
- BN is fused into weights and bias before quantization.
- Bias is applied in requant (`bias_rq`); accumulator bias terms are set to zero.
