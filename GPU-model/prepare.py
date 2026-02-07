import os
import torch
import timm
from datasets import load_dataset
from PIL import Image
from mobile_net import MobileNetV1


# ──────────────────────────────────────────────
# Weight transfer utilities (moved from mobile_net.py)
# ──────────────────────────────────────────────

def _build_weight_mapping():
    """
    Build a mapping from timm's mobilenetv1_100 parameter names
    to our custom MobileNetV1 parameter names.
    timm groups blocks by stride stages; we use flat block_ids matching TF.
    """
    timm_to_custom_blocks = {
        'blocks.0.0': 'conv_dw_1',
        'blocks.1.0': 'conv_dw_2',
        'blocks.1.1': 'conv_dw_3',
        'blocks.2.0': 'conv_dw_4',
        'blocks.2.1': 'conv_dw_5',
        'blocks.3.0': 'conv_dw_6',
        'blocks.3.1': 'conv_dw_7',
        'blocks.3.2': 'conv_dw_8',
        'blocks.3.3': 'conv_dw_9',
        'blocks.3.4': 'conv_dw_10',
        'blocks.3.5': 'conv_dw_11',
        'blocks.4.0': 'conv_dw_12',
        'blocks.4.1': 'conv_dw_13',
    }

    timm_sublayer_to_index = {
        'conv_dw': '0',
        'bn1': '1',
        'conv_pw': '3',
        'bn2': '4',
    }

    mapping = {}

    # Initial conv block
    mapping['conv_stem.weight'] = 'conv1.0.weight'
    for suffix in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
        mapping[f'bn1.{suffix}'] = f'conv1.1.{suffix}'

    # Depthwise separable blocks
    for timm_block, custom_block in timm_to_custom_blocks.items():
        for timm_sub, idx in timm_sublayer_to_index.items():
            if 'conv' in timm_sub:
                mapping[f'{timm_block}.{timm_sub}.weight'] = f'{custom_block}.{idx}.weight'
            else:
                for suffix in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                    mapping[f'{timm_block}.{timm_sub}.{suffix}'] = f'{custom_block}.{idx}.{suffix}'

    return mapping


def transfer_weights(custom_model, official_model):
    """Transfer weights from the timm official model to our custom model."""
    mapping = _build_weight_mapping()
    official_sd = official_model.state_dict()
    custom_sd = custom_model.state_dict()

    transferred = 0
    skipped = []

    for timm_name, custom_name in mapping.items():
        if timm_name in official_sd and custom_name in custom_sd:
            custom_sd[custom_name] = official_sd[timm_name]
            transferred += 1
        else:
            skipped.append((timm_name, custom_name))

    # Classifier: timm uses nn.Linear, we use nn.Conv2d(1x1)
    if 'classifier.weight' in official_sd:
        custom_sd['conv_preds.weight'] = official_sd['classifier.weight'].unsqueeze(-1).unsqueeze(-1)
        transferred += 1
    if 'classifier.bias' in official_sd:
        custom_sd['conv_preds.bias'] = official_sd['classifier.bias']
        transferred += 1

    custom_model.load_state_dict(custom_sd)

    if skipped:
        for t, c in skipped:
            print(f"  - Could not map '{t}' -> '{c}'. Skipping.")

    return transferred


# ──────────────────────────────────────────────
# Dataset download
# ──────────────────────────────────────────────

def download_imagenet_val(output_dir='imagenet_val'):
    """
    Download the ImageNet (ILSVRC2012) validation set via HuggingFace datasets
    and organize it into class-label sub-directories for use with
    torchvision.datasets.ImageFolder.

    Prerequisites (one-time setup):
      1. Accept the dataset terms at:
         https://huggingface.co/datasets/ILSVRC/imagenet-1k
      2. Log in:  huggingface-cli login
    """
    if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"Dataset directory '{output_dir}' already exists. Skipping download.")
        return

    print("Downloading ImageNet validation split from HuggingFace...")
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation")

    os.makedirs(output_dir, exist_ok=True)

    total = len(ds)
    for i, sample in enumerate(ds):
        label = sample["label"]
        class_dir = os.path.join(output_dir, f"{label:04d}")
        os.makedirs(class_dir, exist_ok=True)

        img = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(os.path.join(class_dir, f"{i:06d}.JPEG"))

        if (i + 1) % 5000 == 0 or (i + 1) == total:
            print(f"  Saved {i + 1}/{total} images")

    print(f"Dataset saved to '{output_dir}' ({total} images, {len(os.listdir(output_dir))} classes)")


# ──────────────────────────────────────────────
# Model weight preparation
# ──────────────────────────────────────────────

def prepare_model_weights(save_path='my_mobilenet_with_weights.pth'):
    """Download pretrained weights from timm and save them for our custom model."""
    if os.path.isfile(save_path):
        print(f"Weights file '{save_path}' already exists. Skipping.")
        return

    print("Building custom MobileNet V1 architecture...")
    my_mobilenet = MobileNetV1()

    print("Loading official MobileNet V1 with pre-trained ImageNet weights (timm)...")
    official_mobilenet = timm.create_model('mobilenetv1_100', pretrained=True)
    official_mobilenet.eval()

    print("Transferring weights...")
    num_transferred = transfer_weights(my_mobilenet, official_mobilenet)
    print(f"Weight transfer complete. ({num_transferred} parameter tensors transferred)")

    torch.save(my_mobilenet.state_dict(), save_path)
    print(f"Model weights saved to '{save_path}'")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == '__main__':
    prepare_model_weights()
    download_imagenet_val()
    print("\nDone. You can now run:  python mobile_net.py")
