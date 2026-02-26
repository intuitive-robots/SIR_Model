import torch
from pathlib import Path
from networks.vision_encoder.cnn import CNN, SimpleImageEncoder

import os

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

def load_pretrained_image_encoder(name: str):
    if "256" in name:
        embed_dim = 256
    elif "37" in name:
        embed_dim = 37
    else:
        raise ValueError(f"Unknown embed_dim for pretrained image encoder: {name}")
    
    current_dir = Path(__file__).resolve().parent
    
    path = os.path.join(current_dir, f"../../trained_models/cropped_image_feature/{name}.pth")
    
    if "fusion" in name:
        model = SimpleImageEncoder(6, embed_dim)
    else:
        model = CNN("ResNet18", embed_dim, pretrained=True, in_channels=3)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    
    return model

def crop_and_resize_to_64_for_fusion(img_left: torch.Tensor, img_right: torch.Tensor, 
                                     mask_left: torch.Tensor, mask_right: torch.Tensor = None) -> torch.Tensor:
    """
    Crops the object from both left and right images.
    If mask_right is None, generates a black image for the right view.
    Returns a concatenated 6-channel tensor (assuming 3-channel inputs).
    """
    # 1. Process Left View (Always exists for the left-loop, or is passed as None if handling union logic elsewhere)
    if mask_left is not None:
        crop_left = crop_and_resize_to_64(img_left, mask_left)
    else:
        # Fallback if we ever call this where left is missing (e.g. iterating right list unique objects)
        crop_left = torch.zeros((3, 64, 64), dtype=img_left.dtype, device=img_left.device)

    # 2. Process Right View
    if mask_right is not None:
        crop_right = crop_and_resize_to_64(img_right, mask_right)
    else:
        # Create black image matching left crop's dimensions/type/device
        c, h, w = crop_left.shape
        crop_right = torch.zeros((c, h, w), dtype=img_left.dtype, device=img_left.device)
        
    # 3. Fuse: (3, 64, 64) + (3, 64, 64) -> (6, 64, 64)
    fused_crop = torch.cat((crop_left, crop_right), dim=0)
    
    return fused_crop

def crop_and_resize_to_64(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Crops the masked object from the image (background set to black),
    resizes it to fit within 64x64 using Nearest Neighbor interpolation,
    and centers it on a 64x64 canvas.
    """
    target_size = 64
    offset = 3  # Padding offset
    
    # --- 1. DILATE MASK (PADDING) ---
    # Ensure mask is (N, C, H, W) float for max_pool2d
    if mask.dim() == 2:   # (H, W)
        mask_input = mask.unsqueeze(0).unsqueeze(0).float()
    elif mask.dim() == 3: # (1, H, W) or (C, H, W) - usually mask is 1 channel
        mask_input = mask.unsqueeze(0).float()
    else:
        mask_input = mask.float()

    # MaxPool with stride 1 acts as dilation (expands True regions)
    mask_dilated = F.max_pool2d(
        mask_input, 
        kernel_size=offset * 2 + 1, 
        padding=offset, 
        stride=1
    ).squeeze()

    # --- 2. BLACK OUT BACKGROUND ---
    # Clone to avoid inplace errors if image is part of graph
    image_masked = image.clone() * mask_dilated

    # --- 3. GET BOUNDING BOX OF DILATED MASK ---
    rows, cols = torch.nonzero(mask_dilated, as_tuple=True)
    
    if len(rows) == 0:
        # Return black 64x64 if empty
        c_dim = image.shape[0] if image.dim() == 3 else 1
        return torch.zeros((c_dim, target_size, target_size), dtype=image.dtype, device=image.device)
    
    y_min, y_max = rows.min().item(), rows.max().item()
    x_min, x_max = cols.min().item(), cols.max().item()

    # --- 4. CROP ---
    if image.dim() == 3:
        crop = image_masked[:, y_min:y_max, x_min:x_max]
    else:
        crop = image_masked[y_min:y_max, x_min:x_max].unsqueeze(0)

    # --- 5. RESIZE (NEAREST, MAINTAIN ASPECT RATIO) ---
    # size=63, max_size=64 ensures the longest edge fits in 64, 
    # while the shorter edge scales proportionally.
    crop_resized = TF.resize(
        crop, 
        size=target_size - 1, 
        max_size=target_size, 
        interpolation=InterpolationMode.NEAREST, 
        antialias=True
    )
    
    # --- 6. PAD TO CENTER (64x64) ---
    c, h, w = crop_resized.shape
    canvas = torch.zeros((c, target_size, target_size), dtype=image.dtype, device=image.device)
    
    y_offset = (target_size - h) // 2
    x_offset = (target_size - w) // 2
    
    canvas[:, y_offset:y_offset+h, x_offset:x_offset+w] = crop_resized
    
    return canvas