"""Short package usage notes for generated solution prompts."""

from __future__ import annotations

CORE_CARDS = [
    {
        "title": "PyTorch baseline",
        "content": (
            "Use `torch.utils.data.Dataset` and `DataLoader` for custom data. "
            "Keep the first draft small: simple model, small batch size, and one epoch. "
            "Choose the best available PyTorch device with "
            "`device = torch.device('cuda' if torch.cuda.is_available() else "
            "'mps' if torch.backends.mps.is_available() else 'cpu')`; this supports CUDA, Apple MPS, and CPU. "
            "Pretrained torchvision/timm weights are allowed when useful; do not install new packages. "
            "Always call `model.train()` for training and `model.eval()` with `torch.no_grad()` for validation."
        ),
    },
    {
        "title": "scikit-learn baseline",
        "content": (
            "Use `train_test_split(..., random_state=42, stratify=y if classification labels are available)` "
            "for a quick hold-out split. Use metrics from `sklearn.metrics` and print one numeric validation metric."
        ),
    },
    {
        "title": "Image loading",
        "content": (
            "For PNG/JPEG/TIFF files, prefer `PIL.Image.open(path).convert(...)`, then convert to NumPy or Torch tensors. "
            "Avoid unavailable image libraries such as `cv2` or `skimage`."
        ),
    },
]

MEDICAL_CARDS = [
    {
        "title": "MONAI classification",
        "content": (
            "For medical image classification, prefer `monai.data.ImageDataset` when labels align with image paths. "
            "Use transforms from `monai.transforms`, for example `EnsureChannelFirst`, `ScaleIntensity`, `Resize`, "
            "and `ToTensor`. For dictionary-style datasets, use `EnsureChannelFirstd` and other `*d` transforms."
        ),
    },
    {
        "title": "Medical image readers",
        "content": (
            "Use `nibabel` for NIfTI files, `pydicom` for DICOM metadata/slices, and `SimpleITK` when a robust "
            "medical image reader is needed. Convert loaded images to NumPy arrays, normalize intensities, and keep "
            "spatial sizes small in first drafts."
        ),
    },
    {
        "title": "MONAI model choices",
        "content": (
            "For a conservative first draft, use MONAI networks only when they match the task clearly, such as "
            "`monai.networks.nets.DenseNet121` for classification or `UNet` for segmentation. Keep dimensions "
            "consistent with the data: use 2D models for 2D images and 3D models for volumes."
        ),
    },
]


def package_cards_text(include_medical: bool = True) -> str:
    cards = CORE_CARDS + (MEDICAL_CARDS if include_medical else [])
    return "\n".join(f"- {card['title']}: {card['content']}" for card in cards)
