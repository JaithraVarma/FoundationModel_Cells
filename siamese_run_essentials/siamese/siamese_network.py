import torch
import torch.nn as nn
import torch.nn.functional as F
#\ from torchvision.models import vit_b_16, ViT_B_16_Weights
from siamese.vision_transformer import vit_b_16, ViT_B_16_Weights
from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet50"):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        '''

        super().__init__()

        if backbone not in models.__dict__:
            # self.backbone = Xception()
            self.backbone = vit_b_16(weights=ViT_B_16_Weights, image_size=224)
            print('Using Vision Transformer as backbone')

            # Get the number of features that are outputted by the last layer of backbone network.
            out_features = list(self.backbone.modules())[-1].out_features

            # Create an MLP (multi-layer perceptron) as the classification head.
            # Classifies if provided combined feature vector of the 2 images represent same player or different.
            self.cls_head = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(out_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Dropout(p=0.5),
                nn.Linear(512, 64),
                nn.BatchNorm1d(64),
                nn.Sigmoid(),
                nn.Dropout(p=0.5),

                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        else:
            # Create a backbone network from the pretrained models provided in torchvision.models

            # self.backbone = models.__dict__[backbone](pretrained=True, progress=True)
            # self.backbone = vit_b_16(pretrained=True, progress=True)
            self.backbone = vit_b_16(weights=ViT_B_16_Weights, progress=True, image_size=224)

            print(f'Using Vision {backbone} as backbone')

            # Get the number of features that are outputted by the last layer of backbone network.
            out_features = list(self.backbone.modules())[-1].out_features

            # Create an MLP (multi-layer perceptron) as the classification head.
            # Classifies if provided combined feature vector of the 2 images represent same player or different.
            self.cls_head = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(out_features * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # No sigmoid here
            )

    def forward(self, img1, img2):
        feat1 , _ = self.backbone(img1)  # [B, 768]
        feat2 , _ = self.backbone(img2)  # [B, 768]

        combined_features = torch.cat([feat1, feat2], dim=-1)  # [B, 1536]
        logits = self.cls_head(combined_features)

        return torch.sigmoid(logits), logits

    def save_grads(self, grad):
        '''
        Save gradients of output w.r.t. input for generating saliency maps.

            Parameters:
                    grad (torch.Tensor): shape=[b, num_channels, height, width], Gradient tensor
        '''
        self.backprop_grad = grad


import torch.nn as nn

class SiameseFeatureNetwork(nn.Module):
    def __init__(self, siamese_network):
        '''
        Modifies the provided SiameseNetwork to extract features before the final classification layer.

        Parameters:
            siamese_network (SiameseNetwork): An instance of the SiameseNetwork class.
        '''
        super().__init__()

        # Use the backbone from the provided SiameseNetwork
        self.backbone = siamese_network.backbone

        # Extract the classification head from the provided SiameseNetwork
        # Assuming we want to extract features from the layer before the final classification layer
        # Adjust the index [-2] based on where the last but one layer is, considering your network's architecture
        self.cls_feature_head = nn.Sequential(
            *list(siamese_network.cls_head.children())[:-1]  # Copy all layers except the last one
        )

        self.fc = nn.Sequential(
            *(list(siamese_network.cls_head.children())[-1:])  # Copy only the last layer
        )
    def forward(self, img1, img2):
        '''
        Returns the feature vectors before the final classification layer for two images.

        Parameters:
            img1 (torch.Tensor): shape=[b, 3, 224, 224]
            img2 (torch.Tensor): shape=[b, 3, 224, 224]
        where b = batch size

        Returns:
            feature1 (torch.Tensor): The feature vector of img1.
            feature2 (torch.Tensor): The feature vector of img2.
        '''
        # Process both images through the backbone to get their feature representations
        feat1, _ = self.backbone(img1)
        feat2, _ = self.backbone(img2)

        # Combine the features by element-wise multiplication
        combined_features = torch.cat([feat1, feat2], dim=-1)  # ? Best practice

        # Pass the combined feature vector through the modified classification head
        feature = self.cls_feature_head(combined_features)

        output = self.fc(feature)

        return output, feature