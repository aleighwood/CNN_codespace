#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#if defined(__has_include)
#if __has_include("xparameters.h")
#include "xparameters.h"
#endif
#endif

#include "mobilenet_v1_ctrl_bram_xil.h"

/*
 * Minimal Vitis-side bring-up skeleton for mobilenet_v1_htg_vsl5_top when the
 * feature-map path is exposed as a separate AXI BRAM memory window.
 *
 * Replace the fallback base addresses below with the real XPAR_* macros from
 * your generated xparameters.h once the Vivado block design is fixed.
 */

#ifndef XPAR_MOBILENET_CTRL_BASEADDR
#define XPAR_MOBILENET_CTRL_BASEADDR 0xA0000000u
#endif

#ifndef XPAR_MOBILENET_FM_BRAM_BASEADDR
#define XPAR_MOBILENET_FM_BRAM_BASEADDR 0xA1000000u
#endif

#ifndef MOBILENET_TIMEOUT_POLLS
#define MOBILENET_TIMEOUT_POLLS 200000000u
#endif

#define INPUT_IMAGE_W 224u
#define INPUT_IMAGE_H 224u
#define INPUT_IMAGE_C 3u
#define INPUT_IMAGE_BYTES (INPUT_IMAGE_W * INPUT_IMAGE_H * INPUT_IMAGE_C)
#define DEMO_TILE_COUNT 196u

static void load_demo_input(volatile uint8_t *fm_base)
{
    size_t i;

    for (i = 0; i < INPUT_IMAGE_BYTES; ++i) {
        /*
         * Replace this with your PS preprocessing output tensor.
         */
        fm_base[i] = (uint8_t)(i & 0xffu);
    }
}

static void load_demo_tile_mask(UINTPTR ctrl_base)
{
    static uint8_t all_keep_mask[DEMO_TILE_COUNT];
    size_t i;

    for (i = 0; i < sizeof(all_keep_mask); ++i) {
        all_keep_mask[i] = 1u;
    }

    mobilenet_ctrl_bram_xil_tile_mask_load(ctrl_base,
                                           all_keep_mask,
                                           sizeof(all_keep_mask));
}

static void print_output_probe(volatile int8_t *fm_base)
{
    unsigned int i;

    /*
     * This is only a probe point for first hardware bring-up.
     * Replace this with the final FC output offset once the PL memory map is
     * frozen in Vivado.
     */
    printf("Output probe:");
    for (i = 0; i < 10u; ++i) {
        printf(" %d", (int)fm_base[i]);
    }
    printf("\n");
}

int main(void)
{
    const UINTPTR ctrl_base = (UINTPTR)XPAR_MOBILENET_CTRL_BASEADDR;
    volatile uint8_t *fm_base = (volatile uint8_t *)((UINTPTR)XPAR_MOBILENET_FM_BRAM_BASEADDR);
    int rc;

    printf("MobileNet V1 Vitis bring-up example\n");
    printf("CTRL base = 0x%08lx\n", (unsigned long)ctrl_base);
    printf("FM BRAM   = 0x%08lx\n", (unsigned long)(UINTPTR)fm_base);

    load_demo_input(fm_base);

    mobilenet_ctrl_bram_xil_clear_done(ctrl_base);
    mobilenet_ctrl_bram_xil_set_input_shape(ctrl_base, INPUT_IMAGE_H, INPUT_IMAGE_W);
    mobilenet_ctrl_bram_xil_set_tile_skip(ctrl_base, 1);
    load_demo_tile_mask(ctrl_base);
    mobilenet_ctrl_bram_xil_start(ctrl_base);

    rc = mobilenet_ctrl_bram_xil_wait_done(ctrl_base, MOBILENET_TIMEOUT_POLLS);
    if (rc != 0) {
        printf("ERROR: inference timed out, status=0x%08x\n",
               mobilenet_ctrl_bram_xil_status(ctrl_base));
        return 1;
    }

    printf("Inference done, status=0x%08x\n",
           mobilenet_ctrl_bram_xil_status(ctrl_base));
    print_output_probe((volatile int8_t *)fm_base);
    return 0;
}
