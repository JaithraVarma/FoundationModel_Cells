import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Import from the V2 transformer script
# from src.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from .vision_transformer_v2 import vit_b_16, ViT_B_16_Weights # CORRECTED: Relative import from same dir

from torchvision import models

# Setup logger for this module
logger = logging.getLogger(__name__)
# Optional: Add basic handler if this module might be run standalone or needs output
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)


class SiameseNetwork(nn.Module):
    # Note: Renaming the class here might be safer if both v1 and v2 are imported elsewhere
    # class SiameseNetworkV2(nn.Module):
    def __init__(self, backbone="resnet50", return_attention=False, return_patch_tokens=False):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
                    return_attention (bool): If True, configures the backbone to return attention maps.
                    return_patch_tokens (bool): If True, configures the backbone to return patch tokens.
        '''

        super().__init__()
        self.return_attention = return_attention
        self.return_patch_tokens = return_patch_tokens

        # Note: This logic assumes ViT backbone primarily. Needs adjustment for others.
        if backbone == "vit_b_16": # Be specific for ViT
            # Pass the flags to the V2 backbone
            self.backbone = vit_b_16(
                weights=ViT_B_16_Weights.IMAGENET1K_V1,
                progress=True,
                image_size=224,
                return_attention=self.return_attention,
                return_patch_tokens=self.return_patch_tokens # Pass new flag
            )
            print(f'Using Vision Transformer V2 {backbone} as backbone. Return attention: {self.return_attention}, Return patch tokens: {self.return_patch_tokens}')
            try:
                hidden_dim = self.backbone.hidden_dim
                logger.info(f"Using hidden_dim from backbone: {hidden_dim}")
            except AttributeError:
                logger.warning("Could not find hidden_dim attribute on backbone. Assuming 768 for ViT-B/16.")
                hidden_dim = 768

            self.cls_head = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(hidden_dim * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # No sigmoid here
            )
        else:
            # Fallback for other backbones (unlikely to support new flags)
            logger.warning(f"Backbone {backbone} selected. Attention maps/patch tokens might not be supported/returned.")
            if backbone not in models.__dict__:
                raise ValueError(f"Unsupported backbone: {backbone}")
            # Cannot pass flags to standard torchvision models
            self.backbone = models.__dict__[backbone](pretrained=True, progress=True)
            # Determine out_features for generic torchvision models
            if hasattr(self.backbone, 'fc'):
                out_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
            elif hasattr(self.backbone, 'classifier'):
                 if isinstance(self.backbone.classifier, nn.Linear):
                     out_features = self.backbone.classifier.in_features
                     self.backbone.classifier = nn.Identity()
                 else:
                     logger.error(f"Cannot automatically determine output features or modify classifier for {backbone}. Manual adjustment needed.")
                     out_features = 1000
            else:
                logger.error(f"Cannot determine output features for backbone {backbone}. Assuming 1000.")
                out_features = 1000

            logger.info(f"Using {backbone} as backbone (Attention/Patch tokens unavailable). Deduced out_features: {out_features}")

            self.cls_head = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(out_features * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )


    def get_backbone_outputs(self, img):
        """Passes image through the backbone and returns its raw output(s)."""
        # The V2 backbone now returns based on flags set in __init__
        return self.backbone(img)

    def forward(self, img1, img2):
        """Calculates the similarity between two images."""
        # Use get_backbone_outputs to get potentially rich output
        output1 = self.get_backbone_outputs(img1)
        output2 = self.get_backbone_outputs(img2)

        # Extract CLS features (always the first element)
        feat1 = output1[0] if isinstance(output1, tuple) else output1
        feat2 = output2[0] if isinstance(output2, tuple) else output2

        # Check feature dimensions
        if feat1.dim() != 2 or feat2.dim() != 2:
            logger.error(f"Unexpected feature dimensions from backbone: {feat1.shape}, {feat2.shape}. Expected [B, C].")
            dummy_logits = torch.zeros((img1.shape[0], 1), device=img1.device)
            return torch.sigmoid(dummy_logits), dummy_logits # Return dummy values

        combined_features = torch.cat([feat1, feat2], dim=-1)
        logits = self.cls_head(combined_features)
        return torch.sigmoid(logits), logits # Return probability and logits

    def save_grads(self, grad):
        '''
        Save gradients of output w.r.t. input for generating saliency maps.

            Parameters:
                    grad (torch.Tensor): shape=[b, num_channels, height, width], Gradient tensor
        '''
        self.backprop_grad = grad


# --- Remove or adapt SiameseFeatureNetwork if not needed for V2 ---
# class SiameseFeatureNetwork(nn.Module):
#     ...

