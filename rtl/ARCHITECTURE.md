# RTL Architecture (MobileNet v1 int8)

This document provides a block-level diagram of the RTL architecture and a short legend.

## Top-Level Block Diagram

```mermaid
flowchart TB
  %% External Interfaces
  extfm[(Feature Map Memory)]
  extdw[(DW Tile Buffer Memory)]
  extparam[(Param Write Port)]

  %% Top
  top[mobilenet_v1_top]

  %% Control
  ctrl[mobilenet_v1_ctrl]
  tilectrl[tile_ctrl]

  %% Runners
  conv1[conv1_tile_runner]
  dws[dws_tile_runner]
  gap[gap_runner]
  fc[fc_runner]

  %% Common datapath blocks
  tr[tile_reader]
  tw[tile_writer]
  lb[line_buffer_3x3]
  macv[conv3x3_mac_vec]
  dwc[dw_conv_3x3]
  rq[requant_relu6]
  pwtr[pw_tile_reader]
  pwc[pw_conv_1x1]

  %% Params
  cache[mobilenet_v1_param_cache]

  %% Top wiring
  extfm <-->|rd/wr| top
  extdw <-->|rd/wr| top
  extparam --> top

  top --> ctrl
  ctrl --> tilectrl

  top --> conv1
  top --> dws
  top --> gap
  top --> fc

  top --> cache
  cache --> conv1
  cache --> dws
  cache --> gap
  cache --> fc

  %% Conv1 internal path
  conv1 --> tr --> lb --> macv --> rq --> tw

  %% DWS internal path
  dws --> tr --> lb --> dwc --> rq --> tw
  dws --> pwtr --> pwc --> rq --> tw

  %% GAP + FC internal paths
  gap --> rq --> tw
  fc --> rq --> tw
```

## Dataflow Summary

1. **Top-Level Control**
   - `mobilenet_v1_top` orchestrates the full network: conv1 → 13 depthwise-separable blocks → global average pool → FC.
   - `mobilenet_v1_ctrl` computes layer shapes, stride, and tile coordinates and starts the appropriate runner.

2. **Tiling and Memory**
   - `tile_ctrl` iterates output tiles and computes corresponding input tile ranges (including padding).
   - `tile_reader` streams a tile from planar memory and injects zeros for padding.
   - `tile_writer` writes output tiles back to planar memory.

3. **Conv1 Path (3×3 stride-2)**
   - `tile_reader` → `line_buffer_3x3` → `conv3x3_mac_vec` (parallel OC) → `requant_relu6` → `tile_writer`.
   - Partial sums are accumulated per input channel, then requantized per output channel.

4. **Depthwise + Pointwise Path**
   - **Depthwise**: `tile_reader` → `line_buffer_3x3` → `dw_conv_3x3` → `requant_relu6` → write depthwise tile buffer.
   - **Pointwise**: `pw_tile_reader` streams channel-major DW tiles → `pw_conv_1x1` → `requant_relu6` → `tile_writer`.
   - Pointwise weights are cached in groups via `PW_GROUP` to reduce ROM bandwidth.

5. **Head (GAP + FC)**
   - `gap_runner` sums each channel plane, then requantizes to a single value per channel.
   - `fc_runner` performs a dot product of the GAP vector with FC weights and requantizes outputs.

## Parameter Cache

- `mobilenet_v1_param_cache` stores weights and quant parameters and supports either ROM init (`INIT_*`) or a write port.
- Pointwise weights are prefetched in groups to a small cache, synchronized with `pw_group_req/pw_group_ready`.

## Memory Layout (Assumed)

- Feature maps are **planar**: base + (channel × H×W) + (row × W + col).
- Depthwise tile buffer is **channel-major**: base + (channel × tile_h×tile_w) + (row × tile_w + col).

---

If you prefer an ASCII-only diagram or a more detailed timing/handshake diagram, tell me which format you want.

---

## Detailed Sub-Block Diagrams

### Conv1 Tile Runner (3×3, stride 2)

```mermaid
flowchart LR
  subgraph CONV1[conv1_tile_runner]
    TR[tile_reader] --> LB[line_buffer_3x3]
    LB --> MAC[conv3x3_mac_vec<br/>OC_PAR parallel]
    MAC --> PSUM[(psum_mem\nper-oc group, per-pixel)]
    PSUM --> RQ[requant_relu6]
    RQ --> TW[tile_writer]
  end

  FM[(Feature Map Memory)]
  FM <-->|rd| TR
  TW -->|wr| FM
```

Notes:
- Reads one **input channel at a time**, accumulates into `psum_mem` across channels.
- Outputs one **OC_PAR group** at a time after accumulation and requantization.
- Output tile is written back to planar feature-map memory.

### Depthwise + Pointwise Tile Runner

```mermaid
flowchart LR
  subgraph DWS[dws_tile_runner]
    subgraph DW[Depthwise Path]
      TRd[tile_reader] --> LBd[line_buffer_3x3]
      LBd --> DW3[dw_conv_3x3]
      DW3 --> RQd[requant_relu6]
      RQd --> TWDW[tile_writer<br/>dw tile buf]
    end

    subgraph PW[Pointwise Path]
      PWTR[pw_tile_reader] --> PW1[pw_conv_1x1]
      PW1 --> RQpw[requant_relu6]
      RQpw --> TWpw[tile_writer]
    end
  end

  FM[(Feature Map Memory)]
  DWBUF[(Depthwise Tile Buffer)]

  FM <-->|rd| TRd
  TWDW -->|wr| DWBUF
  PWTR <-->|rd| DWBUF
  TWpw -->|wr| FM
```

Notes:
- Depthwise runs **per input channel**, writes a single-channel output tile to DW buffer.
- Pointwise reads the DW tile **channel-major** and accumulates across channels.
- Pointwise weights are cached in groups (`PW_GROUP`) via the param cache.

---

## Timing / Handshake Views

### Conv1 Tile Runner (per tile)

```mermaid
sequenceDiagram
  autonumber
  participant CTRL as mobilenet_v1_ctrl
  participant C1 as conv1_tile_runner
  participant TR as tile_reader
  participant LB as line_buffer_3x3
  participant MAC as conv3x3_mac_vec
  participant RQ as requant_relu6
  participant TW as tile_writer

  CTRL->>C1: conv1_start + tile config
  C1->>TR: reader_start
  C1->>LB: line_start

  loop For each input channel
    TR-->>LB: in_valid/in_ready stream
    LB-->>MAC: window valid (stride-gated)
    MAC-->>C1: mac_acc_vec (per window)
    C1->>C1: accumulate into psum_mem
  end

  C1->>RQ: stream psum (per pixel, per OC group)
  RQ-->>TW: out_valid/out_ready
  TW-->>CTRL: done (tile)
```

Key handshakes:
- `tile_reader`: `out_valid/out_ready` backpressure.
- `line_buffer_3x3`: `out_valid/out_ready` per window.
- `requant_relu6`: `in_valid/in_ready` + `out_valid/out_ready`.
- `tile_writer`: single-cycle `start`, then consumes data until tile done.

### Depthwise + Pointwise Tile Runner (per tile)

```mermaid
sequenceDiagram
  autonumber
  participant CTRL as mobilenet_v1_ctrl
  participant DWS as dws_tile_runner
  participant TR as tile_reader
  participant DW as depthwise_stage
  participant TWDW as tile_writer (dw buf)
  participant PWTR as pw_tile_reader
  participant PW as pw_conv_1x1
  participant RQ as requant_relu6
  participant TW as tile_writer (out)
  participant PC as param_cache

  CTRL->>DWS: dws_start + tile config

  loop For each input channel (depthwise)
    DWS->>TR: dw_start_pulse
    TR-->>DW: in_valid/in_ready stream
    DW-->>TWDW: out_valid/out_ready
  end

  DWS->>PC: pw_group_req (PW_GROUP weights)
  PC-->>DWS: pw_group_ready

  loop For each output channel
    DWS->>PWTR: pw_start_pulse
    PWTR-->>PW: in_valid/ready + first/last channel flags
    PW-->>RQ: out_valid/out_ready (acc)
    RQ-->>TW: out_valid/out_ready
  end

  TW-->>CTRL: done (tile)
```

Key handshakes:
- `pw_group_req/pw_group_ready` gates pointwise runs by cache load.
- `pw_tile_reader` emits `first_in_ch/last_in_ch` for accumulation reset/commit.
- `tile_writer` runs per output channel with a new `start` pulse.

---

## Tile Reader/Writer Cycle-by-Cycle Example

This is a minimal, illustrative example (not tied to a specific layer):
- Tile size: `tile_in_h=2`, `tile_in_w=3` (6 pixels total)
- `out_ready` stays high (no backpressure)
- The memory returns data in the same cycle as `rd_en` for simplicity

### `tile_reader` (zero-padding skipped here; all pixels in-bounds)

```text
Cycle | start | rd_en | rd_addr | out_valid | out_data | note
------+-------+-------+---------+-----------+----------+------------------------------
  0   |   1   |   1   |   a0    |     0     |    -     | capture cfg, issue first read
  1   |   0   |   1   |   a1    |     1     |   d0     | deliver data0, issue read1
  2   |   0   |   1   |   a2    |     1     |   d1     | deliver data1, issue read2
  3   |   0   |   1   |   a3    |     1     |   d2     | deliver data2, issue read3
  4   |   0   |   1   |   a4    |     1     |   d3     | deliver data3, issue read4
  5   |   0   |   1   |   a5    |     1     |   d4     | deliver data4, issue read5
  6   |   0   |   0   |   -     |     1     |   d5     | deliver last data, done=1
```

Notes:
- `rd_en` asserts once per element; `out_valid` lags by one cycle.
- If an element is out-of-bounds, `rd_en` is not asserted and `out_data=0` is produced instead.

### `tile_writer` (same tile, planar writeback)

```text
Cycle | start | in_valid | in_data | wr_en | wr_addr | note
------+-------+----------+---------+-------+---------+------------------------------
  0   |   1   |    1     |   q0    |   1   |   b0    | capture cfg, write first
  1   |   0   |    1     |   q1    |   1   |   b1    | write next
  2   |   0   |    1     |   q2    |   1   |   b2    | write next
  3   |   0   |    1     |   q3    |   1   |   b3    | write next
  4   |   0   |    1     |   q4    |   1   |   b4    | write next
  5   |   0   |    1     |   q5    |   1   |   b5    | write last, done=1
```

Notes:
- `start` can coincide with the first `in_valid`.
- `wr_addr` increments in raster order within the tile (row-major).
