import torch
from torchvision import transforms
from torchvision.models import vit_b_16
from siamese_network import SiameseNetwork
import torch

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image, resize


class ModifiedViT(torch.nn.Module):
    def __init__(self, original_vit):
        super(ModifiedViT, self).__init__()

        self.vit = original_vit

        out_features = list(self.vit.modules())[-1].out_features
        print(out_features)

    def forward(self, x):
        x, attn_weights = self.vit(x)

        # Return the features and attention weights of the last block
        return x[:, 0], attn_weights

# Load the pretrained ViT model


# Load the pretrained ViT model
checkpoint = torch.load("/nfs/tier1/users/shk35/projects/embryo_witnessing/models/siamese/vit_b_16/interim.v1/v1/vit_b_16_Adam_lr_5e-06_bs_32_e_100/epoch_34.pth")

model = SiameseNetwork(backbone="vit_b_16")

model.load_state_dict(checkpoint['model_state_dict'])

original_vit = model.backbone

print(original_vit)
# Modify the model to return features and attention weights
# modified_vit = ModifiedViT(original_vit)

# Example usage
# img_tensor = torch.rand(1, 3, 224, 224)  # Dummy input image
feed_shape = [3, 224, 224]

transform = transforms.Compose([
    transforms.CenterCrop(size=(int(feed_shape[1] * 1.1), int(feed_shape[2] * 1.1))),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize(feed_shape[1:])
])

img_path = "/nfs/tier1/users/shk35/projects/embryo_witnessing/data/interim.v1//output_9091_3/557.jpg"
image1 = Image.open(img_path).convert("RGB")


img_tensor = transform(image1).float()
img_tensor = img_tensor.unsqueeze(0)

features, attn_weights = original_vit(img_tensor)

print(features.shape)  # Shape of the features
print(len(attn_weights))  # Number of attention weights
# print(attn_weights.shape)  # Shape of the attention weights of the first block



def get_attention_map(attn_weights, layer_idx=-1, head_idx=None, image_size=224):
    """
    Extracts and processes the attention map from a specific layer and head.
    """
    # Select attention weights from a specific layer
    layer_attn = attn_weights[layer_idx]

    # Check if the batch dimension is present
    if layer_attn.dim() == 3:
        layer_attn = layer_attn.unsqueeze(0)  # Add the batch dimension if not present

    print(layer_attn.shape)

    if head_idx is not None:
        # Use attention from a specific head
        attn = layer_attn[0, head_idx, 1:, 1:]  # Exclude the class token
    else:
        # Average across all heads
        attn = layer_attn[0, :, 1:, 1:].mean(0)


    # Reshape to 2D
    attn_size = int(np.sqrt(attn.shape[0]))
    attn_size = attn.shape[0]
    attn = attn.reshape(attn_size, attn_size)

    # Resize to match the original image size
    attn = attn.detach().numpy()
    attn = Image.fromarray(attn)
    attn = resize(attn, [image_size, image_size])
    attn = np.array(attn)

    return attn / np.max(attn)  # Normalize



# Process attention weights (choose layer and head as needed)
attention_map = get_attention_map(attn_weights, layer_idx=-1, head_idx=0, image_size=224)




# Show original image
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image1)
plt.title("Original Image")
plt.axis("off")

# Show attention map
plt.subplot(1, 2, 2)
plt.imshow(image1)
plt.imshow(attention_map, cmap='jet', alpha=0.5)  # Overlay the attention map
plt.title("Attention Map")
plt.axis("off")

plt.show()