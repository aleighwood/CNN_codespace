#ifndef MOBILENET_V1_REGS_H
#define MOBILENET_V1_REGS_H

#include <stdint.h>

/*
 * Bring-up oriented register helper for mobilenet_v1_axi_lite.
 * This uses byte-at-a-time feature-map writes, which is simple but slow.
 * It is suitable for first hardware validation, not final throughput.
 */

#define MOBILENET_REG_CONTROL      0x00u
#define MOBILENET_REG_STATUS       0x04u
#define MOBILENET_REG_CFG_IN_H     0x08u
#define MOBILENET_REG_CFG_IN_W     0x0cu
#define MOBILENET_REG_FM_ADDR      0x10u
#define MOBILENET_REG_FM_WDATA     0x14u
#define MOBILENET_REG_FM_RDATA     0x18u
#define MOBILENET_REG_FM_CMD       0x1cu
#define MOBILENET_REG_MASK_ADDR    0x20u
#define MOBILENET_REG_MASK_DATA    0x24u
#define MOBILENET_REG_MASK_CMD     0x28u
#define MOBILENET_REG_PARAM_SEL    0x30u
#define MOBILENET_REG_PARAM_ADDR   0x34u
#define MOBILENET_REG_PARAM_DATA   0x38u
#define MOBILENET_REG_PARAM_CMD    0x3cu

#define MOBILENET_CONTROL_START        (1u << 0)
#define MOBILENET_CONTROL_TILE_SKIP    (1u << 1)
#define MOBILENET_CONTROL_IRQ_ENABLE   (1u << 2)
#define MOBILENET_CONTROL_CLEAR_DONE   (1u << 4)

#define MOBILENET_STATUS_BUSY          (1u << 0)
#define MOBILENET_STATUS_DONE_STICKY   (1u << 1)
#define MOBILENET_STATUS_DONE_PULSE    (1u << 2)
#define MOBILENET_STATUS_IRQ           (1u << 3)

static inline void mobilenet_write32(uintptr_t base, uint32_t offset, uint32_t value)
{
    volatile uint32_t *reg = (volatile uint32_t *)(base + offset);
    *reg = value;
}

static inline uint32_t mobilenet_read32(uintptr_t base, uint32_t offset)
{
    volatile uint32_t *reg = (volatile uint32_t *)(base + offset);
    return *reg;
}

static inline void mobilenet_set_input_shape(uintptr_t base, uint16_t height, uint16_t width)
{
    mobilenet_write32(base, MOBILENET_REG_CFG_IN_H, (uint32_t)height);
    mobilenet_write32(base, MOBILENET_REG_CFG_IN_W, (uint32_t)width);
}

static inline void mobilenet_set_tile_skip(uintptr_t base, int enable)
{
    uint32_t ctrl;

    ctrl = mobilenet_read32(base, MOBILENET_REG_CONTROL);
    if (enable) {
        ctrl |= MOBILENET_CONTROL_TILE_SKIP;
    } else {
        ctrl &= ~MOBILENET_CONTROL_TILE_SKIP;
    }
    mobilenet_write32(base, MOBILENET_REG_CONTROL, ctrl);
}

static inline void mobilenet_enable_irq(uintptr_t base, int enable)
{
    uint32_t ctrl;

    ctrl = mobilenet_read32(base, MOBILENET_REG_CONTROL);
    if (enable) {
        ctrl |= MOBILENET_CONTROL_IRQ_ENABLE;
    } else {
        ctrl &= ~MOBILENET_CONTROL_IRQ_ENABLE;
    }
    mobilenet_write32(base, MOBILENET_REG_CONTROL, ctrl);
}

static inline void mobilenet_clear_done(uintptr_t base)
{
    uint32_t ctrl;

    ctrl = mobilenet_read32(base, MOBILENET_REG_CONTROL);
    ctrl |= MOBILENET_CONTROL_CLEAR_DONE;
    mobilenet_write32(base, MOBILENET_REG_CONTROL, ctrl);
}

static inline void mobilenet_start(uintptr_t base)
{
    uint32_t ctrl;

    ctrl = mobilenet_read32(base, MOBILENET_REG_CONTROL);
    ctrl |= MOBILENET_CONTROL_START;
    mobilenet_write32(base, MOBILENET_REG_CONTROL, ctrl);
}

static inline uint32_t mobilenet_status(uintptr_t base)
{
    return mobilenet_read32(base, MOBILENET_REG_STATUS);
}

static inline int mobilenet_busy(uintptr_t base)
{
    return (mobilenet_status(base) & MOBILENET_STATUS_BUSY) != 0u;
}

static inline int mobilenet_done(uintptr_t base)
{
    return (mobilenet_status(base) & MOBILENET_STATUS_DONE_STICKY) != 0u;
}

static inline void mobilenet_fm_write_byte(uintptr_t base, uint32_t addr, uint8_t value)
{
    mobilenet_write32(base, MOBILENET_REG_FM_ADDR, addr);
    mobilenet_write32(base, MOBILENET_REG_FM_WDATA, (uint32_t)value);
    mobilenet_write32(base, MOBILENET_REG_FM_CMD, 1u);
}

static inline uint8_t mobilenet_fm_read_byte(uintptr_t base, uint32_t addr)
{
    mobilenet_write32(base, MOBILENET_REG_FM_ADDR, addr);
    return (uint8_t)mobilenet_read32(base, MOBILENET_REG_FM_RDATA);
}

static inline void mobilenet_tile_mask_write(uintptr_t base, uint32_t addr, uint8_t keep)
{
    mobilenet_write32(base, MOBILENET_REG_MASK_ADDR, addr);
    mobilenet_write32(base, MOBILENET_REG_MASK_DATA, (uint32_t)(keep & 1u));
    mobilenet_write32(base, MOBILENET_REG_MASK_CMD, 1u);
}

static inline void mobilenet_param_write(uintptr_t base, uint8_t sel, uint32_t addr, uint32_t data)
{
    mobilenet_write32(base, MOBILENET_REG_PARAM_SEL, (uint32_t)(sel & 0x1fu));
    mobilenet_write32(base, MOBILENET_REG_PARAM_ADDR, addr & 0x000fffffu);
    mobilenet_write32(base, MOBILENET_REG_PARAM_DATA, data);
    mobilenet_write32(base, MOBILENET_REG_PARAM_CMD, 1u);
}

#endif
