import os
import logging
import torch
import torchvision.transforms as T
from PIL import Image
# Make sure siamese module is correctly imported relative to src/models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add src to path
from src.models.siamese_network import SiameseNetwork
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
from collections import defaultdict

# Import the PairDataset class from the second file
from pair_dataset import PairDataset

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- Visualization Function from Reference ---
def create_single_visualization(image_tensor, attn_grid, patch_size, img_size=224):
    """
    Creates a BGR image array with attention map overlaid as discrete tiles.
    Args:
        image_tensor (torch.Tensor): Original image tensor (C, H, W), normalized, on CPU.
        attn_grid (torch.Tensor): Attention map grid (H_grid, W_grid), UNINTERPOLATED, on CPU.
        patch_size (int): The size of each patch (e.g., 16).
        img_size (int): Target size for the image.
    Returns:
        np.ndarray | None: The BGR image array (H, W, C) uint8 with overlay, or None if error.
    """
    try:
        # 1. Denormalize image and convert to OpenCV format (H, W, C) uint8
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = image_tensor * std + mean
        img = img.clamp(0, 1)
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        raw_bgr = img_bgr.copy() # Keep raw image

        # 2. Process Attention Grid for Tiled Visualization
        attn_np = attn_grid.numpy() # Shape: (H_grid, W_grid)
        # Normalize attention values to 0-1 for colormap
        if attn_np.max() > attn_np.min(): # Avoid division by zero if attention is uniform
             attn_normalized = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8) # Added epsilon
        else:
             attn_normalized = np.zeros_like(attn_np)

        H_grid, W_grid = attn_normalized.shape

        # Create a blank image for the heatmap overlay
        heatmap_overlay = np.zeros_like(img_bgr, dtype=np.uint8)
        cmap = plt.get_cmap('jet') # Get the JET colormap function

        # Draw colored tiles
        for r in range(H_grid):
            for c in range(W_grid):
                color_rgba = cmap(attn_normalized[r, c]) # Get RGBA color (0-1 scale)
                color_bgr = (int(color_rgba[2]*255), int(color_rgba[1]*255), int(color_rgba[0]*255)) # Convert to BGR uint8

                start_row, start_col = r * patch_size, c * patch_size
                end_row, end_col = min(start_row + patch_size, img_size), min(start_col + patch_size, img_size)
                cv2.rectangle(heatmap_overlay, (start_col, start_row), (end_col, end_row), color_bgr, -1)

        # 3. Blend the tiled heatmap with the original image
        alpha = 0.5 # Transparency factor
        overlay_img_bgr = cv2.addWeighted(img_bgr, alpha, heatmap_overlay, 1 - alpha, 0)

        return raw_bgr, overlay_img_bgr # Return both raw and overlay

    except Exception as e:
        logger.error(f"Error creating single visualization: {e}")
        return None, None # Return None for both

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Testing Siamese Network and Visualizing Attention with PairDataset")
    parser = argparse.ArgumentParser()

    # Updated arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        help="Path to the root data directory containing class folders.",
        required=False,
        default='/nfs/tier3/projects/Cell_Model_Data/data/processed/processed_2D_CTC/train/'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        help="Path to the CSV file containing dataset information.",
        required=False,
        default = "/nfs/tier3/projects/Cell_Model_Data/data/processed/processed_2D_CTC/train/train.csv"
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        help="Path to the trained model checkpoint (.pth file).",
        required=False,
        default='/nfs/tier2/users/sm1367/Cell_Model/outputs/vit_b_16_Adam_lr_0.0001_bs_16_e_100_simclr_lambda1_1.0_lambda2_1.0_test_7/best.pth'
    )
    parser.add_argument(
        '-b',
        '--backbone',
        type=str,
        help="Network backbone used in the trained siamese network (must match checkpoint).",
        default="vit_b_16",
        required=False
    )
    parser.add_argument(
        '--img_size',
        type=int,
        help="Image size used during training (for resizing attention maps).",
        required=False,
        default=224
    )
    parser.add_argument(
        '--target_class',
        type=str,
        help="Target class for filtering pairs (if specified).",
        required=False
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help="Whether to shuffle the pairs (default: False for deterministic testing)",
        default=False
    )
    parser.add_argument(
        '--num_pairs',
        type=int,
        help="Number of pairs to visualize (default: 5)",
        required=False,
        default=5
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        help="Number of workers for multiprocessing in pair generation",
        required=False,
        default=4
    )

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- Load PairDataset with existing CSV ---
    try:
        # Verify CSV exists
        if not os.path.isfile(args.csv_path):
            logger.error(f"CSV file not found at: {args.csv_path}")
            exit(1)
            
        dataset = PairDataset(
            path=args.csv_path,
            shuffle_pairs=args.shuffle,
            augment=False,  # No augmentation for visualization
            num_workers=args.num_workers
        )
        logger.info(f"PairDataset loaded successfully with {len(dataset)} pairs")
    except Exception as e:
        logger.error(f"Error loading PairDataset: {e}")
        exit(1)

    # --- Load Model ---
    # logger.info(f"Loading model with backbone: {args.backbone}")
    # try:
    #     model = SiameseNetwork(backbone=args.backbone)
    #     logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
        
    #     checkpoint = torch.load(args.checkpoint_path, map_location=device)
    #     if 'siamese_state_dict' in checkpoint:
    #         model.load_state_dict(checkpoint['siamese_state_dict'])
    #         logger.info("Successfully loaded siamese_state_dict from checkpoint.")
    #     elif 'model_state_dict' in checkpoint:
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         logger.info("Successfully loaded model_state_dict from checkpoint.")
    #     else:
    #         logger.error(f"Could not find expected state dict keys in checkpoint. Available keys: {checkpoint.keys()}")
    #         exit(1)

    #     # Compatibility check
    #     if 'backbone' in checkpoint and checkpoint['backbone'] != args.backbone:
    #         logger.warning(f"Checkpoint backbone ('{checkpoint['backbone']}') differs from specified backbone ('{args.backbone}')!")

    #     # model.load_state_dict(checkpoint['model_state_dict'])
    #     model.to(device)
    #     model.eval()
    #     logger.info("Model loaded successfully.")
    #     patch_size = model.backbone.patch_size
    #     logger.info(f"Model patch size: {patch_size}")
    # except FileNotFoundError:
    #     logger.error(f"Checkpoint file not found at: {args.checkpoint_path}")
    #     exit(1)
    # except AttributeError:
    #     logger.error("Could not determine patch_size from model.backbone. Ensure backbone has patch_size attribute.")
    #     exit(1)
    # except Exception as e:
    #     logger.error(f"Error loading checkpoint or model: {e}")
    #     exit(1)
    logger.info(f"Loading model with backbone: {args.backbone}")
    try:
        model = SiameseNetwork(backbone=args.backbone)
        logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
        
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # Determine the state dictionary to use
        if 'siamese_state_dict' in checkpoint:
            state_dict = checkpoint['siamese_state_dict']
            logger.info("Found siamese_state_dict in checkpoint.")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logger.info("Found model_state_dict in checkpoint.")
        else:
            logger.error(f"Could not find expected state dict keys in checkpoint. Available keys: {checkpoint.keys()}")
            exit(1)

        # Strip 'module.' prefix from state dictionary keys if present
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '') if key.startswith('module.') else key
            new_state_dict[new_key] = value

        # Load the modified state dictionary
        model.load_state_dict(new_state_dict)
        logger.info("Successfully loaded state dictionary into model.")

        # Compatibility check
        if 'backbone' in checkpoint and checkpoint['backbone'] != args.backbone:
            logger.warning(f"Checkpoint backbone ('{checkpoint['backbone']}') differs from specified backbone ('{args.backbone}')!")

        model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")
        patch_size = model.backbone.patch_size
        logger.info(f"Model patch size: {patch_size}")
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at: {args.checkpoint_path}")
        exit(1)
    except AttributeError:
        logger.error("Could not determine patch_size from model.backbone. Ensure backbone has patch_size attribute.")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading checkpoint or model: {e}")
        exit(1)

    # --- Process and visualize pairs from dataset ---
    num_pairs = min(args.num_pairs, len(dataset))
    logger.info(f"Visualizing {num_pairs} image pairs")

    # Filter pairs by target class if specified
    target_class_idx = []
    if args.target_class:
        logger.info(f"Filtering for pairs with target class: {args.target_class}")
        for i in range(len(dataset)):
            (_, _), _, (class1, class2), _ = dataset[i]
            if class1 == args.target_class or class2 == args.target_class:
                target_class_idx.append(i)
        
        if target_class_idx:
            logger.info(f"Found {len(target_class_idx)} pairs with target class {args.target_class}")
            # Select from filtered indices
            indices_to_use = target_class_idx[:num_pairs]
        else:
            logger.warning(f"No pairs found with target class {args.target_class}, using first {num_pairs} pairs")
            indices_to_use = list(range(num_pairs))
    else:
        indices_to_use = list(range(num_pairs))

    # --- Inference and Visualization Loop ---
    for idx, sample_idx in enumerate(indices_to_use):
        try:
            # Get sample from dataset
            (img1_tensor, img2_tensor), true_label, (class1, class2), (path1, path2) = dataset[sample_idx]
            
            # Move tensors to device
            img1_tensor = img1_tensor.unsqueeze(0).to(device)
            img2_tensor = img2_tensor.unsqueeze(0).to(device)
            
            pair_type = "Positive" if true_label.item() == 1 else "Negative"
            logger.info(f"\n--- Processing {pair_type} Pair {idx+1}/{len(indices_to_use)}: {os.path.basename(path1)} vs {os.path.basename(path2)} ---")
            logger.info(f"Classes: {class1} vs {class2}")
            
            raw1, vis1 = None, None
            raw2, vis2 = None, None
            batch_idx = 0

            # --- Get Features, Attention Maps, and Prediction ---
            with torch.no_grad():
                features1, attn_list1 = model.backbone(img1_tensor)
                features2, attn_list2 = model.backbone(img2_tensor)

                # Calculate Prediction
                if features1 is not None and features2 is not None:
                    combined_features = torch.cat([features1, features2], dim=-1)
                    logits = model.cls_head(combined_features)
                    probability = torch.sigmoid(logits)
                    pred_prob = probability[batch_idx].item()
                    pred_prob_text = f"Pred (Sim): {pred_prob:.4f}"
                    logger.info(pred_prob_text)
                else:
                    pred_prob_text = "Pred: N/A"
                    logger.warning(f"Could not get features for pair {idx+1}, cannot calculate prediction.")

            # --- Process and Visualize Attention ---
            # Process Image 1
            if attn_list1:
                last_layer_attn1 = attn_list1[-1][batch_idx].cpu()
                num_patches1 = last_layer_attn1.shape[0] - 1
                if num_patches1 > 0:
                    attn_cls_to_patches1 = last_layer_attn1[0, 1:]
                    H_attn = W_attn = int(math.sqrt(num_patches1))
                    if H_attn * W_attn == num_patches1:
                        attn_grid1 = attn_cls_to_patches1.reshape(H_attn, W_attn)
                        raw1, vis1 = create_single_visualization(img1_tensor[batch_idx].cpu(), attn_grid1, patch_size, args.img_size)
                    else:
                        logger.warning(f"Cannot reshape patch attention (num_patches={num_patches1}) for img1 of pair {idx+1}.")
                else:
                    logger.warning(f"No patches found in attention map for img1 of pair {idx+1}.")
            else:
                logger.warning(f"No attention maps found in attn_list1 for pair {idx+1}.")

            # Process Image 2
            if attn_list2:
                last_layer_attn2 = attn_list2[-1][batch_idx].cpu()
                num_patches2 = last_layer_attn2.shape[0] - 1
                if num_patches2 > 0:
                    attn_cls_to_patches2 = last_layer_attn2[0, 1:]
                    H_attn = W_attn = int(math.sqrt(num_patches2))
                    if H_attn * W_attn == num_patches2:
                        attn_grid2 = attn_cls_to_patches2.reshape(H_attn, W_attn)
                        raw2, vis2 = create_single_visualization(img2_tensor[batch_idx].cpu(), attn_grid2, patch_size, args.img_size)
                    else:
                        logger.warning(f"Cannot reshape patch attention (num_patches={num_patches2}) for img2 of pair {idx+1}.")
                else:
                    logger.warning(f"No patches found in attention map for img2 of pair {idx+1}.")
            else:
                logger.warning(f"No attention maps found in attn_list2 for pair {idx+1}.")

            # Display side-by-side if all components were created
            if raw1 is not None and vis1 is not None and raw2 is not None and vis2 is not None:
                combined_vis = np.hstack((raw1, vis1, raw2, vis2))

                # Add Prediction and True Label Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_color = (255, 255, 255)
                bg_color = (0, 0, 0)
                line_type = 2
                text_y_pred = 30
                text_y_true = 55
                text_x = 10

                true_label_text = f"True: {pair_type}"

                # Prediction text with background
                (pred_width, pred_height), pred_baseline = cv2.getTextSize(pred_prob_text, font, font_scale, line_type)
                cv2.rectangle(combined_vis, (text_x, text_y_pred - pred_height - pred_baseline), (text_x + pred_width, text_y_pred + pred_baseline), bg_color, -1)
                cv2.putText(combined_vis, pred_prob_text, (text_x, text_y_pred), font, font_scale, font_color, line_type)

                # True label text with background
                (true_width, true_height), true_baseline = cv2.getTextSize(true_label_text, font, font_scale, line_type)
                cv2.rectangle(combined_vis, (text_x, text_y_true - true_height - true_baseline), (text_x + true_width, text_y_true + true_baseline), bg_color, -1)
                cv2.putText(combined_vis, true_label_text, (text_x, text_y_true), font, font_scale, font_color, line_type)

                window_title = f"Pair {idx+1} ({pair_type}): {os.path.basename(path1)} vs {os.path.basename(path2)}"
                try:
                    # Replace the cv2.imshow and cv2.waitKey calls with:
                    output_dir = "/nfs/tier2/users/sm1367/Cell_Model/visualization/attention_outputs/test3_siamese_simclr"
                    output_path = os.path.join(output_dir, f"pair_{idx+1}_{pair_type}_{os.path.basename(path1)}_vs_{os.path.basename(path2)}.png")
                    cv2.imwrite(output_path, combined_vis)
                    logger.info(f"Saved visualization to: {output_path}")
                except Exception as display_error:
                    logger.error(f"Failed to display image in window '{window_title}'. Ensure GUI environment: {display_error}")
            else:
                logger.warning(f"Could not create all required visualization components for pair {idx+1}. Skipping display.")

        except Exception as e:
            logger.error(f"Error processing pair {idx+1}: {e}")
            continue

    logger.info("\nTest visualization finished.")