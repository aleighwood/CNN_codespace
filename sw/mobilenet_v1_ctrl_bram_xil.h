#ifndef MOBILENET_V1_CTRL_BRAM_XIL_H
#define MOBILENET_V1_CTRL_BRAM_XIL_H

#include <stddef.h>
#include <stdint.h>

#include "mobilenet_v1_ctrl_bram_regs.h"

/*
 * Vitis-oriented register helper for mobilenet_v1_axi_ctrl_bram.
 * This uses Xil_In32 / Xil_Out32 for the AXI-Lite control window.
 *
 * Define MOBILENET_STANDALONE_STUB for local host-side syntax checks when the
 * Xilinx headers are not available.
 */

#ifdef MOBILENET_STANDALONE_STUB
typedef uintptr_t UINTPTR;

static inline void Xil_Out32(UINTPTR addr, uint32_t value)
{
    volatile uint32_t *reg = (volatile uint32_t *)addr;

    *reg = value;
}

static inline uint32_t Xil_In32(UINTPTR addr)
{
    volatile uint32_t *reg = (volatile uint32_t *)addr;

    return *reg;
}
#else
#include "xil_io.h"
#include "xil_types.h"
#endif

static inline void mobilenet_ctrl_bram_xil_write32(UINTPTR base,
                                                   uint32_t offset,
                                                   uint32_t value)
{
    Xil_Out32(base + (UINTPTR)offset, value);
}

static inline uint32_t mobilenet_ctrl_bram_xil_read32(UINTPTR base,
                                                      uint32_t offset)
{
    return Xil_In32(base + (UINTPTR)offset);
}

static inline void mobilenet_ctrl_bram_xil_set_input_shape(UINTPTR base,
                                                           uint16_t height,
                                                           uint16_t width)
{
    mobilenet_ctrl_bram_xil_write32(base, MOBILENET_CTRL_BRAM_REG_CFG_IN_H,
                                    (uint32_t)height);
    mobilenet_ctrl_bram_xil_write32(base, MOBILENET_CTRL_BRAM_REG_CFG_IN_W,
                                    (uint32_t)width);
}

static inline uint32_t mobilenet_ctrl_bram_xil_status(UINTPTR base)
{
    return mobilenet_ctrl_bram_xil_read32(base, MOBILENET_CTRL_BRAM_REG_STATUS);
}

static inline int mobilenet_ctrl_bram_xil_done(UINTPTR base)
{
    return (mobilenet_ctrl_bram_xil_status(base) &
            MOBILENET_CTRL_BRAM_STATUS_DONE_STICKY) != 0u;
}

static inline uint32_t mobilenet_ctrl_bram_xil_control_read(UINTPTR base)
{
    return mobilenet_ctrl_bram_xil_read32(base, MOBILENET_CTRL_BRAM_REG_CONTROL);
}

static inline void mobilenet_ctrl_bram_xil_control_write(UINTPTR base,
                                                         uint32_t control)
{
    mobilenet_ctrl_bram_xil_write32(base, MOBILENET_CTRL_BRAM_REG_CONTROL,
                                    control);
}

static inline void mobilenet_ctrl_bram_xil_set_tile_skip(UINTPTR base,
                                                         int enable)
{
    uint32_t ctrl = mobilenet_ctrl_bram_xil_control_read(base);

    if (enable) {
        ctrl |= MOBILENET_CTRL_BRAM_CONTROL_TILE_SKIP;
    } else {
        ctrl &= ~MOBILENET_CTRL_BRAM_CONTROL_TILE_SKIP;
    }

    mobilenet_ctrl_bram_xil_control_write(base, ctrl);
}

static inline void mobilenet_ctrl_bram_xil_clear_done(UINTPTR base)
{
    uint32_t ctrl = mobilenet_ctrl_bram_xil_control_read(base);

    ctrl |= MOBILENET_CTRL_BRAM_CONTROL_CLEAR_DONE;
    mobilenet_ctrl_bram_xil_control_write(base, ctrl);
}

static inline void mobilenet_ctrl_bram_xil_start(UINTPTR base)
{
    uint32_t ctrl = mobilenet_ctrl_bram_xil_control_read(base);

    ctrl |= MOBILENET_CTRL_BRAM_CONTROL_START;
    mobilenet_ctrl_bram_xil_control_write(base, ctrl);
}

static inline void mobilenet_ctrl_bram_xil_tile_mask_write(UINTPTR base,
                                                           uint32_t addr,
                                                           uint8_t keep)
{
    mobilenet_ctrl_bram_xil_write32(base, MOBILENET_CTRL_BRAM_REG_MASK_ADDR, addr);
    mobilenet_ctrl_bram_xil_write32(base, MOBILENET_CTRL_BRAM_REG_MASK_DATA,
                                    (uint32_t)(keep & 1u));
    mobilenet_ctrl_bram_xil_write32(base, MOBILENET_CTRL_BRAM_REG_MASK_CMD, 1u);
}

static inline void mobilenet_ctrl_bram_xil_tile_mask_load(UINTPTR base,
                                                          const uint8_t *mask,
                                                          size_t count)
{
    size_t i;

    for (i = 0; i < count; ++i) {
        mobilenet_ctrl_bram_xil_tile_mask_write(base, (uint32_t)i, mask[i]);
    }
}

static inline int mobilenet_ctrl_bram_xil_wait_done(UINTPTR base,
                                                    uint32_t max_poll_iters)
{
    uint32_t i;

    for (i = 0; i < max_poll_iters; ++i) {
        if (mobilenet_ctrl_bram_xil_done(base)) {
            return 0;
        }
    }

    return -1;
}

#endif
