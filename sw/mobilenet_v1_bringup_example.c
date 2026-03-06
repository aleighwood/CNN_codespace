#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "mobilenet_v1_ctrl_bram_regs.h"

/*
 * Minimal PS-side bring-up flow for mobilenet_v1_axi_ctrl_bram.
 *
 * This is a software skeleton for Vitis:
 * - AXI-Lite is used for control/status only.
 * - The feature-map buffer is assumed to be separately memory-mapped from the PS
 *   (for example through an AXI BRAM Controller or a BRAM window).
 *
 * Replace the base addresses below with the addresses exported by Vivado.
 * Replace the demo tensor/mask contents with real preprocessing output.
 */

#ifndef MOBILENET_CTRL_BASEADDR
#define MOBILENET_CTRL_BASEADDR 0xA0000000u
#endif

#ifndef MOBILENET_FM_BASEADDR
#define MOBILENET_FM_BASEADDR   0xA1000000u
#endif

#ifndef MOBILENET_TIMEOUT_POLLS
#define MOBILENET_TIMEOUT_POLLS 200000000u
#endif

#define INPUT_IMAGE_W 224u
#define INPUT_IMAGE_H 224u
#define INPUT_IMAGE_C 3u
#define INPUT_IMAGE_BYTES (INPUT_IMAGE_W * INPUT_IMAGE_H * INPUT_IMAGE_C)

static void load_demo_input(volatile uint8_t *fm_base)
{
    size_t i;

    for (i = 0; i < INPUT_IMAGE_BYTES; ++i) {
        /*
         * Demo content only. In the real system, write the int8 tensor produced
         * by your PS preprocessing flow into this memory window.
         */
        fm_base[i] = (uint8_t)(i & 0xffu);
    }
}

static void load_demo_tile_mask(uintptr_t ctrl_base)
{
    /*
     * Keep every tile in this example. Replace with the ROI-derived mask from
     * your PS preprocessing pipeline.
     */
    static uint8_t all_keep_mask[196];
    size_t i;

    for (i = 0; i < sizeof(all_keep_mask); ++i) {
        all_keep_mask[i] = 1u;
    }

    mobilenet_ctrl_bram_tile_mask_load(ctrl_base,
                                       all_keep_mask,
                                       sizeof(all_keep_mask));
}

static void print_fc_top5(volatile int8_t *fm_base)
{
    /*
     * The current RTL writes FC outputs into the shared feature-map space.
     * For bring-up, this example only dumps the first 10 bytes as a quick
     * visibility check. Replace this with the correct FC output base once you
     * freeze the final PL memory map in Vivado.
     */
    unsigned int i;

    printf("First 10 output bytes:");
    for (i = 0; i < 10u; ++i) {
        printf(" %d", (int)fm_base[i]);
    }
    printf("\n");
}

int main(void)
{
    uintptr_t ctrl_base = (uintptr_t)MOBILENET_CTRL_BASEADDR;
    volatile uint8_t *fm_base = (volatile uint8_t *)((uintptr_t)MOBILENET_FM_BASEADDR);
    int rc;

    printf("MobileNet V1 bring-up example\n");
    printf("CTRL base = 0x%08lx\n", (unsigned long)ctrl_base);
    printf("FM base   = 0x%08lx\n", (unsigned long)(uintptr_t)fm_base);

    load_demo_input(fm_base);

    mobilenet_ctrl_bram_clear_done(ctrl_base);
    mobilenet_ctrl_bram_set_input_shape(ctrl_base, INPUT_IMAGE_H, INPUT_IMAGE_W);
    mobilenet_ctrl_bram_set_tile_skip(ctrl_base, 1);

    load_demo_tile_mask(ctrl_base);

    mobilenet_ctrl_bram_start(ctrl_base);

    rc = mobilenet_ctrl_bram_wait_done(ctrl_base, MOBILENET_TIMEOUT_POLLS);
    if (rc != 0) {
        printf("ERROR: inference timed out, status=0x%08x\n",
               mobilenet_ctrl_bram_status(ctrl_base));
        return 1;
    }

    printf("Inference done, status=0x%08x\n", mobilenet_ctrl_bram_status(ctrl_base));
    print_fc_top5((volatile int8_t *)fm_base);
    return 0;
}
