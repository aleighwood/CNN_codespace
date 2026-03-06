#ifndef MOBILENET_V1_CTRL_BRAM_REGS_H
#define MOBILENET_V1_CTRL_BRAM_REGS_H

#include <stddef.h>
#include <stdint.h>

/*
 * Bring-up oriented register helper for mobilenet_v1_axi_ctrl_bram.
 * Use AXI-Lite for control/status only. Bulk feature-map traffic should go
 * through a separate BRAM or memory window mapped by the PS.
 */

#define MOBILENET_CTRL_BRAM_REG_CONTROL      0x00u
#define MOBILENET_CTRL_BRAM_REG_STATUS       0x04u
#define MOBILENET_CTRL_BRAM_REG_CFG_IN_H     0x08u
#define MOBILENET_CTRL_BRAM_REG_CFG_IN_W     0x0cu
#define MOBILENET_CTRL_BRAM_REG_MASK_ADDR    0x20u
#define MOBILENET_CTRL_BRAM_REG_MASK_DATA    0x24u
#define MOBILENET_CTRL_BRAM_REG_MASK_CMD     0x28u
#define MOBILENET_CTRL_BRAM_REG_PARAM_SEL    0x30u
#define MOBILENET_CTRL_BRAM_REG_PARAM_ADDR   0x34u
#define MOBILENET_CTRL_BRAM_REG_PARAM_DATA   0x38u
#define MOBILENET_CTRL_BRAM_REG_PARAM_CMD    0x3cu

#define MOBILENET_CTRL_BRAM_CONTROL_START        (1u << 0)
#define MOBILENET_CTRL_BRAM_CONTROL_TILE_SKIP    (1u << 1)
#define MOBILENET_CTRL_BRAM_CONTROL_IRQ_ENABLE   (1u << 2)
#define MOBILENET_CTRL_BRAM_CONTROL_CLEAR_DONE   (1u << 4)

#define MOBILENET_CTRL_BRAM_STATUS_BUSY          (1u << 0)
#define MOBILENET_CTRL_BRAM_STATUS_DONE_STICKY   (1u << 1)
#define MOBILENET_CTRL_BRAM_STATUS_DONE_PULSE    (1u << 2)
#define MOBILENET_CTRL_BRAM_STATUS_IRQ           (1u << 3)

static inline void mobilenet_ctrl_bram_write32(uintptr_t base,
                                               uint32_t offset,
                                               uint32_t value)
{
    volatile uint32_t *reg = (volatile uint32_t *)(base + offset);
    *reg = value;
}

static inline uint32_t mobilenet_ctrl_bram_read32(uintptr_t base, uint32_t offset)
{
    volatile uint32_t *reg = (volatile uint32_t *)(base + offset);
    return *reg;
}

static inline void mobilenet_ctrl_bram_set_input_shape(uintptr_t base,
                                                       uint16_t height,
                                                       uint16_t width)
{
    mobilenet_ctrl_bram_write32(base, MOBILENET_CTRL_BRAM_REG_CFG_IN_H,
                                (uint32_t)height);
    mobilenet_ctrl_bram_write32(base, MOBILENET_CTRL_BRAM_REG_CFG_IN_W,
                                (uint32_t)width);
}

static inline uint32_t mobilenet_ctrl_bram_status(uintptr_t base)
{
    return mobilenet_ctrl_bram_read32(base, MOBILENET_CTRL_BRAM_REG_STATUS);
}

static inline int mobilenet_ctrl_bram_busy(uintptr_t base)
{
    return (mobilenet_ctrl_bram_status(base) &
            MOBILENET_CTRL_BRAM_STATUS_BUSY) != 0u;
}

static inline int mobilenet_ctrl_bram_done(uintptr_t base)
{
    return (mobilenet_ctrl_bram_status(base) &
            MOBILENET_CTRL_BRAM_STATUS_DONE_STICKY) != 0u;
}

static inline uint32_t mobilenet_ctrl_bram_control_read(uintptr_t base)
{
    return mobilenet_ctrl_bram_read32(base, MOBILENET_CTRL_BRAM_REG_CONTROL);
}

static inline void mobilenet_ctrl_bram_control_write(uintptr_t base,
                                                     uint32_t control)
{
    mobilenet_ctrl_bram_write32(base, MOBILENET_CTRL_BRAM_REG_CONTROL, control);
}

static inline void mobilenet_ctrl_bram_set_tile_skip(uintptr_t base, int enable)
{
    uint32_t ctrl = mobilenet_ctrl_bram_control_read(base);

    if (enable) {
        ctrl |= MOBILENET_CTRL_BRAM_CONTROL_TILE_SKIP;
    } else {
        ctrl &= ~MOBILENET_CTRL_BRAM_CONTROL_TILE_SKIP;
    }

    mobilenet_ctrl_bram_control_write(base, ctrl);
}

static inline void mobilenet_ctrl_bram_enable_irq(uintptr_t base, int enable)
{
    uint32_t ctrl = mobilenet_ctrl_bram_control_read(base);

    if (enable) {
        ctrl |= MOBILENET_CTRL_BRAM_CONTROL_IRQ_ENABLE;
    } else {
        ctrl &= ~MOBILENET_CTRL_BRAM_CONTROL_IRQ_ENABLE;
    }

    mobilenet_ctrl_bram_control_write(base, ctrl);
}

static inline void mobilenet_ctrl_bram_clear_done(uintptr_t base)
{
    uint32_t ctrl = mobilenet_ctrl_bram_control_read(base);

    ctrl |= MOBILENET_CTRL_BRAM_CONTROL_CLEAR_DONE;
    mobilenet_ctrl_bram_control_write(base, ctrl);
}

static inline void mobilenet_ctrl_bram_start(uintptr_t base)
{
    uint32_t ctrl = mobilenet_ctrl_bram_control_read(base);

    ctrl |= MOBILENET_CTRL_BRAM_CONTROL_START;
    mobilenet_ctrl_bram_control_write(base, ctrl);
}

static inline void mobilenet_ctrl_bram_tile_mask_write(uintptr_t base,
                                                       uint32_t addr,
                                                       uint8_t keep)
{
    mobilenet_ctrl_bram_write32(base, MOBILENET_CTRL_BRAM_REG_MASK_ADDR, addr);
    mobilenet_ctrl_bram_write32(base, MOBILENET_CTRL_BRAM_REG_MASK_DATA,
                                (uint32_t)(keep & 1u));
    mobilenet_ctrl_bram_write32(base, MOBILENET_CTRL_BRAM_REG_MASK_CMD, 1u);
}

static inline void mobilenet_ctrl_bram_tile_mask_load(uintptr_t base,
                                                      const uint8_t *mask,
                                                      size_t count)
{
    size_t i;

    for (i = 0; i < count; ++i) {
        mobilenet_ctrl_bram_tile_mask_write(base, (uint32_t)i, mask[i]);
    }
}

static inline void mobilenet_ctrl_bram_param_write(uintptr_t base,
                                                   uint8_t sel,
                                                   uint32_t addr,
                                                   uint32_t data)
{
    mobilenet_ctrl_bram_write32(base, MOBILENET_CTRL_BRAM_REG_PARAM_SEL,
                                (uint32_t)(sel & 0x1fu));
    mobilenet_ctrl_bram_write32(base, MOBILENET_CTRL_BRAM_REG_PARAM_ADDR,
                                addr & 0x000fffffu);
    mobilenet_ctrl_bram_write32(base, MOBILENET_CTRL_BRAM_REG_PARAM_DATA, data);
    mobilenet_ctrl_bram_write32(base, MOBILENET_CTRL_BRAM_REG_PARAM_CMD, 1u);
}

static inline int mobilenet_ctrl_bram_wait_done(uintptr_t base,
                                                uint32_t max_poll_iters)
{
    uint32_t i;

    for (i = 0; i < max_poll_iters; ++i) {
        if (mobilenet_ctrl_bram_done(base)) {
            return 0;
        }
    }

    return -1;
}

#endif
