# Scripts README

This directory contains Python utilities for preprocessing, golden-model generation, ROI/tile-skip analysis, and accuracy checks.

**Setup**

Most scripts assume the `mobilenet_env` conda environment and that you run them from the repo root. Example:

```
conda run -n mobilenet_env python scripts/image_to_input_mem.py --image dog.jpg --input-shape 224,224,3
```

## Script Index

| Script | Purpose | Example |
| --- | --- | --- |
| `scripts/add_weights.py` | Download MobileNet weights and save to `mobilenet_imagenet.weights.h5`. | `conda run -n mobilenet_env python scripts/add_weights.py` |
| `scripts/build_imagenet_npz.py` | Build NPZ dataset from TFDS ImageNet for quantization benchmarking. | `conda run -n mobilenet_env python scripts/build_imagenet_npz.py --output imagenet_val.npz --num-samples 500 --manual-dir /path/to/imagenet` |
| `scripts/compare_all_layers.py` | Compare per-layer `.mem` outputs (RTL vs golden). | `python scripts/compare_all_layers.py --mem-dir rtl/mem` |
| `scripts/compare_fc.py` | Compare FC int8 output (`fc_expected.mem` vs `fc_out_hw.mem`). | `python scripts/compare_fc.py --expected rtl/mem/fc_expected.mem --actual rtl/mem/fc_out_hw.mem` |
| `scripts/compare_fc_logits.py` | Compare FC int32 logits (`fc_logits_expected.mem` vs `fc_logits_hw.mem`). | `python scripts/compare_fc_logits.py --expected rtl/mem/fc_logits_expected.mem --actual rtl/mem/fc_logits_hw.mem` |
| `scripts/compare_int8_paths.py` | Compare FP32 vs TFLite int8 vs golden int8 outputs. | `conda run -n mobilenet_env python scripts/compare_int8_paths.py --val-dir ILSVRC2012_val --num-samples 3` |
| `scripts/compare_models.py` | Quick TF vs NumPy forward sanity check (small classes). | `conda run -n mobilenet_env python scripts/compare_models.py` |
| `scripts/eval_golden_int8.py` | Evaluate golden int8 model on ILSVRC2012 val subset. | `conda run -n mobilenet_env python scripts/eval_golden_int8.py --val-dir ILSVRC2012_val --num-samples 50` |
| `scripts/eval_imagenet_val.py` | Evaluate MobileNet on ILSVRC2012 val set (FP32 or TFLite). | `conda run -n mobilenet_env python scripts/eval_imagenet_val.py --val-dir ILSVRC2012_val --gt-file ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt --tflite quantized_models/mobilenet_int8.tflite --top5` |
| `scripts/eval_roi_tile_skip.py` | Evaluate ROI tile-skip golden model on val subset. | `conda run -n mobilenet_env python scripts/eval_roi_tile_skip.py --val-dir ILSVRC2012_val --gt-file ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt --tflite quantized_models/mobilenet_int8.tflite --num-samples 20 --tile-size 16 --roi-halo-tiles 1 --roi-halo-layers 1` |
| `scripts/export_mobilenet_int8_mem.py` | Export MobileNet int8 weights/biases and requant params to `.mem` files. | `conda run -n mobilenet_env python scripts/export_mobilenet_int8_mem.py --output-dir rtl/mem --tflite quantized_models/mobilenet_int8_ilsvrc2012_5000.tflite --use-tflite-weights` |
| `scripts/gen_golden_fc.py` | Generate golden FC outputs for a given input mem. | `conda run -n mobilenet_env python scripts/gen_golden_fc.py --input-shape 224,224,3 --input-mem-in rtl/mem/input_rand.mem --input-mem rtl/mem/input_rand.mem --expected-mem rtl/mem/fc_expected.mem --expected-logits-mem rtl/mem/fc_logits_expected.mem --tflite quantized_models/mobilenet_int8.tflite --q31` |
| `scripts/gen_tile_mask_mem.py` | Generate per-layer tile mask memory from ROI bitmap. | `conda run -n mobilenet_env python scripts/gen_tile_mask_mem.py --mask-npy processed_frames/dog/dog_mask.npy --input-shape 224,224,3 --tile-size 16 --roi-halo-tiles 1 --roi-halo-layers 3 --out-mem rtl/mem/tile_mask.mem` |
| `scripts/image_crop.py` | Tiled window discovery demo from mask or real image. | `python scripts/image_crop.py --image dog.jpg --image-size 224 --window-width 16 --window-height 16 --output-image rtl/mem/tiles.png --overlay-rgb` |
| `scripts/image_to_input_mem.py` | Convert RGB image to int8 CHW `.mem` input. | `conda run -n mobilenet_env python scripts/image_to_input_mem.py --image dog.jpg --input-shape 224,224,3 --output-mem rtl/mem/input_rand.mem --tflite quantized_models/mobilenet_int8.tflite` |
| `scripts/image_to_input_mem_roi.py` | Convert an ROI region to int8 CHW `.mem` input. | `conda run -n mobilenet_env python scripts/image_to_input_mem_roi.py --image dog.jpg --roi 32,32,128,128 --input-shape 224,224,3 --output-mem rtl/mem/input_roi.mem` |
| `scripts/mobilenet_test.py` | NumPy MobileNet sanity test with random weights. | `python scripts/mobilenet_test.py` |
| `scripts/not_in_use.py` | Old scratch experiments (not used). | `python scripts/not_in_use.py` |
| `scripts/object_crop.py` | Background removal + mask generation for images/videos. | `conda run -n mobilenet_env python scripts/object_crop.py dog.jpg --keep-size --mask-size 224 --mask-thresh 10 --bg-color 0` |
| `scripts/quantization_benchmark.py` | Benchmark quantization variants on NPZ dataset. | `conda run -n mobilenet_env python scripts/quantization_benchmark.py --npz imagenet_val.npz --weights mobilenet_imagenet.weights.h5` |
| `scripts/quantize_from_imagenet_val.py` | Quantize MobileNet using ILSVRC2012 val images. | `conda run -n mobilenet_env python scripts/quantize_from_imagenet_val.py --val-dir ILSVRC2012_val --output quantized_models/mobilenet_int8_ilsvrc2012.tflite --num-samples 5000` |
| `scripts/roi_compute_savings.py` | Estimate compute savings from ROI tile skipping on one image. | `python scripts/roi_compute_savings.py --image dog.jpg --tile-size 16 --roi-halo-tiles 1 --roi-halo-layers 1` |
| `scripts/roi_preview.py` | Preview ROI masks/bboxes on val images. | `python scripts/roi_preview.py --val-dir ILSVRC2012_val --num 6 --output-dir rtl/mem/roi_preview` |
| `scripts/roi_tile_overlay.py` | Draw bitmap tile overlay on image. | `python scripts/roi_tile_overlay.py --image dog.jpg --tile-size 16 --halo-tiles 1 --out rtl/mem/roi_tile_overlay.png` |
| `scripts/roi_visualize.py` | Visualize ROI bbox and bitmap mask. | `python scripts/roi_visualize.py --image dog.jpg --image-size 224 --out-bbox rtl/mem/roi_bbox_vis.png --out-mask rtl/mem/roi_bitmap_vis.png` |
| `scripts/run_roi_golden.py` | Generate ROI input mem + golden outputs for one image. | `conda run -n mobilenet_env python scripts/run_roi_golden.py --image dog.jpg --input-shape 224,224,3 --roi-mode tile-skip --auto --tile-size 16 --roi-halo-tiles 1 --roi-halo-layers 1` |
| `scripts/sweep_bitmap.py` | Sweep bitmap tile-skip settings and report savings + top‑5. | `conda run -n mobilenet_env python scripts/sweep_bitmap.py --image dog.jpg --input-shape 224,224,3 --tflite quantized_models/mobilenet_int8.tflite --tile-size 16 --halo-tiles 0 1 --halo-layers 0 3` |

If you want a shorter index (only the ones you actively use), tell me which scripts to keep and I can slim this down.*** End Patch"}Дополнитель to=functions.apply_patch log. code to=functions.apply_patch ／久久낌assistant to=functions.apply_patch
