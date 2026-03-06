PS-side bring-up helpers

Files
- `sw/mobilenet_v1_regs.h`: Minimal register-access helper for `mobilenet_v1_axi_lite`.
- `sw/mobilenet_v1_ctrl_bram_regs.h`: Register helper for `mobilenet_v1_axi_ctrl_bram`, intended for AXI-Lite control with a separate BRAM/memory datapath for feature-map traffic.
- `sw/mobilenet_v1_bringup_example.c`: Minimal Vitis-side software skeleton for loading a demo tensor, loading a tile mask, starting inference, and polling for completion.
- `sw/mobilenet_v1_ctrl_bram_xil.h`: Vitis-oriented helper that uses `Xil_In32` / `Xil_Out32` for the AXI-Lite control window.
- `sw/mobilenet_v1_vitis_bringup_example.c`: Vitis app skeleton assuming the feature-map buffer is exposed as a separate AXI BRAM window.

Notes
- `mobilenet_v1_regs.h` is the legacy bring-up path that writes one feature-map byte at a time through AXI-Lite. It is useful for very early validation but too slow for normal frame loading.
- `mobilenet_v1_ctrl_bram_regs.h` is the preferred helper for the newer `mobilenet_v1_axi_ctrl_bram` RTL wrapper.
- `mobilenet_v1_ctrl_bram_xil.h` is the Vitis-specific version of that helper. It is the one to use once you have a Vivado design and generated `xparameters.h`.
- For the `mobilenet_v1_axi_ctrl_bram` path:
  - use AXI-Lite for control/status, tile-mask writes, and optional parameter pokes
  - map the feature-map buffer through a separate BRAM or memory window from the PS
- `mobilenet_v1_bringup_example.c` is a software skeleton, not a finished application:
  - replace the placeholder base addresses with the values exported by Vivado/Vitis
  - replace the demo tensor and all-keep tile mask with real preprocessing outputs
  - replace the placeholder output readback with the final FC output buffer location once the PL memory map is frozen
- `mobilenet_v1_vitis_bringup_example.c` assumes:
  - the control block has an AXI-Lite base address
  - the feature-map buffer is mapped through a separate AXI BRAM Controller window
  - you will replace the fallback `XPAR_*` macros with the actual generated names/values from `xparameters.h`
