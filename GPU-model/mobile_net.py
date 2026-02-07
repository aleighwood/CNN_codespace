import torch
import torch.nn as nn
from torchvision import datasets, transforms


def _conv_block(in_channels, out_channels, kernel_size, stride):
    """A standard convolution block with Batch Norm and ReLU6."""
    padding = (kernel_size - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


def _depthwise_separable_conv_block(in_channels, out_channels, stride):
    """A depthwise separable convolution block: DW Conv + BN + ReLU6 + PW Conv + BN + ReLU6."""
    return nn.Sequential(
        # Depthwise convolution
        nn.Conv2d(in_channels, in_channels, 3,
                  stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        # Pointwise convolution
        nn.Conv2d(in_channels, out_channels, 1,
                  stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class MobileNetV1(nn.Module):
    """
    MobileNet V1 architecture in PyTorch.
    Equivalent to the Keras Functional API version in MobileNet_tf.py.
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        # Initial Convolution Block
        self.conv1 = _conv_block(3, 32, 3, stride=2)

        # Depthwise Separable Blocks (matching TF block_ids 1-13)
        self.conv_dw_1 = _depthwise_separable_conv_block(32, 64, stride=1)
        self.conv_dw_2 = _depthwise_separable_conv_block(64, 128, stride=2)
        self.conv_dw_3 = _depthwise_separable_conv_block(128, 128, stride=1)
        self.conv_dw_4 = _depthwise_separable_conv_block(128, 256, stride=2)
        self.conv_dw_5 = _depthwise_separable_conv_block(256, 256, stride=1)
        self.conv_dw_6 = _depthwise_separable_conv_block(256, 512, stride=2)
        self.conv_dw_7 = _depthwise_separable_conv_block(512, 512, stride=1)
        self.conv_dw_8 = _depthwise_separable_conv_block(512, 512, stride=1)
        self.conv_dw_9 = _depthwise_separable_conv_block(512, 512, stride=1)
        self.conv_dw_10 = _depthwise_separable_conv_block(512, 512, stride=1)
        self.conv_dw_11 = _depthwise_separable_conv_block(512, 512, stride=1)
        self.conv_dw_12 = _depthwise_separable_conv_block(512, 1024, stride=2)
        self.conv_dw_13 = _depthwise_separable_conv_block(1024, 1024, stride=1)

        # Classification Head
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.001)
        self.conv_preds = nn.Conv2d(1024, num_classes, 1)

    def forward(self, x):
        x = self.conv1(x)

        x = self.conv_dw_1(x)
        x = self.conv_dw_2(x)
        x = self.conv_dw_3(x)
        x = self.conv_dw_4(x)
        x = self.conv_dw_5(x)
        x = self.conv_dw_6(x)
        x = self.conv_dw_7(x)
        x = self.conv_dw_8(x)
        x = self.conv_dw_9(x)
        x = self.conv_dw_10(x)
        x = self.conv_dw_11(x)
        x = self.conv_dw_12(x)
        x = self.conv_dw_13(x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.conv_preds(x)
        x = x.view(x.size(0), -1)
        x = torch.softmax(x, dim=1)
        return x


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate(model, data_loader, device):
    """Run inference on the dataset and return top-1 and top-5 accuracy."""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # Top-1
            _, pred_top1 = outputs.max(dim=1)
            correct_top1 += pred_top1.eq(labels).sum().item()

            # Top-5
            _, pred_top5 = outputs.topk(5, dim=1)
            correct_top5 += pred_top5.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(data_loader)}  "
                      f"running top-1: {100 * correct_top1 / total:.2f}%  "
                      f"top-5: {100 * correct_top5 / total:.2f}%")

    top1_acc = 100 * correct_top1 / total
    top5_acc = 100 * correct_top5 / total
    return top1_acc, top5_acc


if __name__ == '__main__':
    weights_path = 'my_mobilenet_with_weights.pth'
    data_dir = 'imagenet_val'
    batch_size = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load model with pretrained weights
    print("Loading MobileNet V1 weights...")
    model = MobileNetV1()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # 2. Load prepared ImageNet validation dataset
    print(f"Loading dataset from '{data_dir}'...")
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
    )
    print(f"Dataset: {len(val_dataset)} images, {len(val_dataset.classes)} classes")

    # 3. Evaluate
    print("Running inference...")
    top1, top5 = evaluate(model, val_loader, device)
    print(f"\nResults on ImageNet validation set:")
    print(f"  Top-1 accuracy: {top1:.2f}%")
    print(f"  Top-5 accuracy: {top5:.2f}%")
