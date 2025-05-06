import torch
import torch.nn as nn
import torch.nn.functional as F
#\ from torchvision.models import vit_b_16, ViT_B_16_Weights
from .vision_transformer import vit_b_16, ViT_B_16_Weights # Adjusted import
from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet50"):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        '''

        super().__init__()

        # Check if the requested backbone is a standard torchvision model OR if it's our specific ViT
        # Note: torchvision 0.8.2 (from requirements) does NOT include vit_b_16. 
        # So, if backbone is 'vit_b_16', it won't be in models.__dict__.
        if backbone == "vit_b_16" or backbone not in models.__dict__:
            # Use the local Vision Transformer implementation
            print(f'Using local Vision Transformer ({backbone}) as backbone')
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT, image_size=224) # Use .DEFAULT weights enum
            
            # --- REVERTED: Keep the original ViT head --- 
            # Determine the output features of the standard ViT head (should be 1000 for ImageNet)
            try:
                # Get the output features of the final layer in the heads block
                module_list = list(self.backbone.heads.modules())
                last_linear = [m for m in module_list if isinstance(m, nn.Linear)][-1]
                out_features = last_linear.out_features # Output features of the final FC layer (e.g., 1000)
                print(f"Detected backbone's final layer output features: {out_features}")
            except (AttributeError, IndexError, TypeError):
                 print("Warning: Could not automatically determine backbone output features. Assuming 1000 for ViT ImageNet head.")
                 out_features = 1000 
            # --------------------------------------------

            # Define the classification head to accept concatenated features (out_features * 2)
            self.cls_head = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(out_features * 2, 512), # Input should now be 2000 (1000*2)
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Dropout(p=0.5),
                nn.Linear(512, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(), 
                nn.Dropout(p=0.5),

                nn.Linear(64, 1)
            )
            print(f"cls_head defined to accept {out_features * 2} input features.")

        else:
            # This block handles standard torchvision models (e.g., resnet50)
            print(f'Using torchvision {backbone} as backbone')
            self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

            # Get the number of features from the standard model's classifier layer (e.g., fc)
            if hasattr(self.backbone, 'fc'): # Common for ResNets
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity() # Remove the original classifier
            elif hasattr(self.backbone, 'classifier'): # Common for VGG, DenseNet
                 # More complex - might need more specific handling depending on model
                 # Example for DenseNet:
                 if isinstance(self.backbone.classifier, nn.Linear):
                     in_features = self.backbone.classifier.in_features
                     self.backbone.classifier = nn.Identity()
                 else: 
                     print(f"Warning: Classifier structure for {backbone} not automatically handled. Using Identity.")
                     in_features = list(self.backbone.modules())[-1].in_features # Guess based on last linear layer
                     self.backbone.classifier = nn.Identity() # Attempt removal
            else:
                # Fallback: try to get features from the second to last module assuming it's linear
                try:
                    in_features = list(self.backbone.modules())[-2].out_features
                    # We don't necessarily remove the layer here, just get the features size
                except (IndexError, AttributeError, TypeError):
                     print(f"Warning: Could not determine output features for {backbone}. Cannot build cls_head.")
                     in_features = 0 # Indicate failure
            
            out_features = in_features
            print(f"Determined backbone output features: {out_features}")

            # Create an MLP classification head using the structure from the 'else' block in original code
            if out_features > 0:
                self.cls_head = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(out_features * 2, 512), # Takes concatenated features
                    nn.ReLU(),
                    nn.Linear(512, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)  # No sigmoid here
                )
            else:
                self.cls_head = nn.Identity() # Cannot build head


    def forward(self, img1, img2):
        # Pass images through the backbone
        # Assuming backbone returns features directly, or (features, aux_outputs)
        output1 = self.backbone(img1) 
        output2 = self.backbone(img2)

        # Extract features (handle tuple output if necessary)
        feat1 = output1[0] if isinstance(output1, tuple) else output1
        feat2 = output2[0] if isinstance(output2, tuple) else output2
        
        # --- DEBUG: Print feature shapes --- REMOVED
        # print(f"DEBUG: feat1 shape: {feat1.shape}, feat2 shape: {feat2.shape}") 
        # -------------------------------------

        combined_features = torch.cat([feat1, feat2], dim=-1)
        
        # --- DEBUG: Print combined feature shape --- REMOVED
        # print(f"DEBUG: combined_features shape: {combined_features.shape}") 
        # -------------------------------------------
        
        logits = self.cls_head(combined_features)

        # Return sigmoid probabilities (for evaluation/prediction) and raw logits (for loss)
        return torch.sigmoid(logits), logits

    def save_grads(self, grad):
        '''
        Save gradients of output w.r.t. input for generating saliency maps.

            Parameters:
                    grad (torch.Tensor): shape=[b, num_channels, height, width], Gradient tensor
        '''
        self.backprop_grad = grad


# --- SiameseFeatureNetwork (modified for clarity and robustness) ---

class SiameseFeatureNetwork(nn.Module):
    def __init__(self, siamese_network: SiameseNetwork):
        '''
        Modifies the provided SiameseNetwork to extract features before the final classification layer.

        Parameters:
            siamese_network (SiameseNetwork): An instance of the SiameseNetwork class.
        '''
        super().__init__()

        # Use the backbone from the provided SiameseNetwork
        self.backbone = siamese_network.backbone

        # Extract the classification head layers EXCEPT the final one
        cls_head_children = list(siamese_network.cls_head.children())
        if len(cls_head_children) > 1:
            self.cls_feature_head = nn.Sequential(*cls_head_children[:-1])
            self.fc = nn.Sequential(*cls_head_children[-1:]) # Final layer
        elif len(cls_head_children) == 1:
            # If cls_head is just one layer, feature head is Identity
            self.cls_feature_head = nn.Identity()
            self.fc = nn.Sequential(*cls_head_children)
        else:
            # Handle case where cls_head might be empty or Identity
            self.cls_feature_head = nn.Identity()
            self.fc = nn.Identity()
        

    def forward(self, img1, img2):
        '''
        Returns the final output logits and the feature vectors before the final classification layer.

        Parameters:
            img1 (torch.Tensor): shape=[b, 3, 224, 224]
            img2 (torch.Tensor): shape=[b, 3, 224, 224]
        where b = batch size

        Returns:
            output (torch.Tensor): The final logits from the classification head.
            feature (torch.Tensor): The feature vector before the final classification layer.
        '''
        # Process both images through the backbone
        output1 = self.backbone(img1)
        output2 = self.backbone(img2)

        # Extract features
        feat1 = output1[0] if isinstance(output1, tuple) else output1
        feat2 = output2[0] if isinstance(output2, tuple) else output2

        # Combine the features 
        combined_features = torch.cat([feat1, feat2], dim=-1)

        # Pass the combined feature vector through the layers before the final one
        feature = self.cls_feature_head(combined_features)

        # Pass the intermediate feature through the final layer
        output = self.fc(feature)

        return output, feature 