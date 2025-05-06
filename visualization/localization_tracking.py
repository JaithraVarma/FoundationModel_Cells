# src/models/localization_tracking.py

import os
import pandas as pd # Keep for potential future use
import logging
import torch
import torchvision.transforms as T
from PIL import Image
import sys
# Ensure src directory is in path to import sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Adjust the import based on your actual file structure if siamese_network.py is elsewhere
try:
    from src.features.siamese_network_v2 import SiameseNetwork # Use relative import
except ImportError:
    print("Error importing SiameseNetwork from V2. Make sure siamese_network_v2.py is in the same directory (src/models) and src is in PYTHONPATH.")
    print(f"Current sys.path: {sys.path}h")
    sys.exit(1)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import random
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import skimage.measure
import skimage.transform
import skimage.draw # For drawing lines

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers: # Avoid adding multiple handlers if script is re-run
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Visualization Helper ---
def create_single_visualization(image_tensor, attn_grid, patch_size, img_size=224):
    """
    Creates a BGR image array with attention map overlaid as discrete tiles.
    Returns both the raw BGR image and the overlay image.
    """
    try:
        # 1. Denormalize image and convert to OpenCV format (H, W, C) uint8
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = image_tensor.cpu() * std + mean # Ensure tensor is on CPU
        img = img.clamp(0, 1)
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        raw_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Raw BGR image

        # 2. Process Attention Grid for Tiled Visualization
        attn_np = attn_grid.cpu().numpy() # Ensure tensor is on CPU
        if attn_np.max() > attn_np.min():
             attn_normalized = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
        else:
             attn_normalized = np.zeros_like(attn_np)

        H_grid, W_grid = attn_normalized.shape
        heatmap_overlay = np.zeros_like(raw_bgr, dtype=np.uint8)
        cmap = plt.get_cmap('jet')

        for r in range(H_grid):
            for c in range(W_grid):
                color_rgba = cmap(attn_normalized[r, c])
                color_bgr = (int(color_rgba[2]*255), int(color_rgba[1]*255), int(color_rgba[0]*255))
                start_row, start_col = r * patch_size, c * patch_size
                end_row, end_col = min(start_row + patch_size, img_size), min(start_col + patch_size, img_size)
                cv2.rectangle(heatmap_overlay, (start_col, start_row), (end_col, end_row), color_bgr, -1)

        # 3. Blend the tiled heatmap
        alpha = 0.5
        overlay_img_bgr = cv2.addWeighted(raw_bgr.copy(), alpha, heatmap_overlay, 1 - alpha, 0)

        return raw_bgr, overlay_img_bgr # Return both

    except Exception as e:
        logger.error(f"Error creating single visualization: {e}")
        return None, None

# --- Data Handling Helpers ---
def scan_and_prepare_data(data_dir):
    logger.info(f"Scanning data directory: {data_dir}")
    class_data = defaultdict(list)
    found_images = 0
    try:
        if not os.path.isdir(data_dir):
            logger.error(f"Data directory not found: {data_dir}")
            return None
        for class_folder in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_folder)
            if os.path.isdir(class_path):
                images_path = os.path.join(class_path, 'images')
                if os.path.isdir(images_path):
                    for image_file in os.listdir(images_path):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')): # More robust extension check
                            try:
                                parts = image_file.split('_')
                                if len(parts) >= 4:
                                    frame_num_str = parts[-3]
                                    frame_num = int(frame_num_str)
                                    full_image_path = os.path.join(images_path, image_file)
                                    class_data[class_folder].append((frame_num, full_image_path))
                                    found_images += 1
                                else:
                                    logger.warning(f"Skipping {image_file} in {class_folder}: Cannot extract frame number (not enough parts).")
                            except (ValueError, IndexError):
                                logger.warning(f"Skipping {image_file} in {class_folder}: Error parsing frame number.")
        if not class_data:
            logger.error(f"No valid class folders with images found in {data_dir}.")
            return None
        logger.info(f"Found {found_images} images across {len(class_data)} classes.")
        for class_name in class_data:
            class_data[class_name].sort(key=lambda x: x[0])
        return class_data
    except Exception as e:
        logger.error(f"Error scanning directory {data_dir}: {e}")
        return None

def generate_pairs(class_data):
    pairs = []
    positive_pairs_generated = 0
    negative_pairs_generated = 0
    positive_class_found = False
    for class_name, images in class_data.items():
        if len(images) >= 2:
            # logger.info(f"Generating positive pairs from class: {class_name}")
            for i in range(len(images) - 1):
                _, path1 = images[i]
                _, path2 = images[i+1]
                pairs.append((path1, path2, 1))
                positive_pairs_generated += 1
            positive_class_found = True
            # break # Consider generating positives from all eligible classes? For now, just the first.
    if not positive_class_found:
        logger.warning("Could not find any class with >= 2 images to generate positive pairs.")

    eligible_classes = [cls for cls, imgs in class_data.items() if len(imgs) > 0]
    if len(eligible_classes) >= 2:
        # Generate one negative pair for simplicity
        class_a_name = eligible_classes[0]
        class_b_name = eligible_classes[1]
        # logger.info(f"Generating negative pair from classes: {class_a_name} and {class_b_name}")
        _, path_a = class_data[class_a_name][0]
        _, path_b = class_data[class_b_name][0]
        pairs.append((path_a, path_b, 0))
        negative_pairs_generated += 1
        # Can add more negative pairs later if needed
    else:
        logger.warning("Could not find two different classes with images to generate a negative pair.")

    logger.info(f"Generated {positive_pairs_generated} positive pairs and {negative_pairs_generated} negative pairs.")
    random.shuffle(pairs) # Shuffle for variety during visualization
    return pairs

# --- Step 4: Localization Helper ---
def localize_cells_from_attention(attn_grid, target_size, patch_size, threshold_quantile=0.7, min_area=10):
    """
    Localizes potential cells based on attention map using thresholding and connected components.
    Args:
        attn_grid (np.ndarray): Attention map grid (H_grid, W_grid) from CLS to patches, on CPU.
        target_size (int): The size of the original image (e.g., 224).
        patch_size (int): The size of patches (e.g., 16).
        threshold_quantile (float): Quantile for thresholding the attention map.
        min_area (int): Minimum area in pixels for a connected component to be considered a cell.
    Returns:
        tuple: (list of bounding boxes [(x, y, w, h), ...], list of centroids [(cx, cy), ...], binary_mask)
    """
    boxes = []
    centroids = []
    binary_mask = np.zeros((target_size, target_size), dtype=np.uint8) # Default empty mask
    try:
        # 1. Upscale attention map
        attn_map_resized = skimage.transform.resize(
            attn_grid, (target_size, target_size), order=1, preserve_range=True
        )
        # 2. Smooth (optional)
        attn_map_smoothed = cv2.GaussianBlur(attn_map_resized, (5, 5), 0)
        # 3. Thresholding
        if np.ptp(attn_map_smoothed) > 1e-6: # Check if map is not flat
            threshold_value = np.quantile(attn_map_smoothed, threshold_quantile)
            binary_mask = (attn_map_smoothed > threshold_value).astype(np.uint8)
        else:
            logger.debug("Attention map is flat, skipping thresholding.")
            return boxes, centroids, binary_mask # Return empty if flat

        # 4. Find Connected Components using skimage.measure
        labels = skimage.measure.label(binary_mask > 0, connectivity=2, background=0)
        properties = skimage.measure.regionprops(labels, intensity_image=attn_map_smoothed)

        # 5. Filter components and get bounding boxes/centroids
        for prop in properties:
            if prop.area >= min_area:
                minr, minc, maxr, maxc = prop.bbox
                bx, by = minc, minr
                bw, bh = maxc - minc, maxr - minr
                boxes.append((bx, by, bw, bh))
                # Use weighted centroid for potentially better localization within the blob
                centroid_y, centroid_x = prop.weighted_centroid
                centroids.append((int(centroid_x), int(centroid_y))) # Store as (x, y)

        # logger.debug(f"Found {len(boxes)} components after filtering.")

    except Exception as e:
        logger.error(f"Error during cell localization: {e}", exc_info=True) # Log traceback

    return boxes, centroids, (binary_mask * 255) # Return mask for potential visualization

# --- Step 5: Tracking Helpers ---
def extract_cell_embeddings(patch_tokens, centroids, patch_size, H_grid, W_grid):
    """
    Extracts patch embeddings corresponding to cell centroids.
    Requires patch_tokens tensor to be on CPU.
    """
    if patch_tokens is None:
        logger.error("Patch tokens are None. Cannot extract cell embeddings.")
        return []
    if patch_tokens.dim() == 3: # Handle potential batch dim
        patch_tokens = patch_tokens.squeeze(0)

    cell_embeddings = []
    num_patches = H_grid * W_grid
    # Check if number of tokens matches grid size (N vs H*W)
    if patch_tokens.shape[0] != num_patches:
        logger.error(f"Mismatch: patch_tokens has {patch_tokens.shape[0]} tokens, expected {num_patches} ({H_grid}x{W_grid}).")
        return cell_embeddings

    for cx, cy in centroids:
        patch_r = min(max(0, cy // patch_size), H_grid - 1)
        patch_c = min(max(0, cx // patch_size), W_grid - 1)
        patch_idx = patch_r * W_grid + patch_c

        if 0 <= patch_idx < patch_tokens.shape[0]:
            cell_embeddings.append(patch_tokens[patch_idx])
        else:
            logger.warning(f"Centroid ({cx}, {cy}) -> patch index {patch_idx} out of bounds ({patch_tokens.shape[0]}). Skipping.")

    return cell_embeddings

def match_cells_hungarian(embeddings1, embeddings2, similarity_threshold=0.5):
    """ Matches cells using cosine similarity and Hungarian algorithm. Requires embeddings on CPU. """
    matches = []
    if not embeddings1 or not embeddings2:
        return matches

    try:
        emb1_tensor = torch.stack(embeddings1).detach().numpy() # Ensure numpy on CPU
        emb2_tensor = torch.stack(embeddings2).detach().numpy()

        if emb1_tensor.ndim != 2 or emb2_tensor.ndim != 2:
             logger.error(f"Invalid embedding dimensions for matching: {emb1_tensor.ndim}, {emb2_tensor.ndim}")
             return matches

        similarity_matrix = cosine_similarity(emb1_tensor, emb2_tensor)
        cost_matrix = 1.0 - similarity_matrix

        # Check for NaN/Inf in cost matrix
        if np.isnan(cost_matrix).any() or np.isinf(cost_matrix).any():
            logger.warning("NaN or Inf detected in cost matrix. Attempting to replace with large values.")
            cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=1.0) # Replace bad values

        # Handle empty cost matrix case
        if cost_matrix.size == 0:
            logger.debug("Empty cost matrix, no matches possible.")
            return matches

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
             # Check bounds before accessing similarity_matrix
             if r < similarity_matrix.shape[0] and c < similarity_matrix.shape[1]:
                 similarity = similarity_matrix[r, c]
                 if similarity >= similarity_threshold:
                     matches.append((r, c))
             else:
                 logger.warning(f"Index out of bounds during match filtering: ({r}, {c}) vs matrix shape {similarity_matrix.shape}")

    except ValueError as e:
        logger.error(f"ValueError during Hungarian matching: {e}. Cost matrix shape: {cost_matrix.shape}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during cell matching: {e}", exc_info=True)

    # logger.debug(f"Found {len(matches)} matches above threshold {similarity_threshold}.")
    return matches

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting Localization and Tracking Visualization")
    parser = argparse.ArgumentParser(description="Run Siamese Network inference, localize cells using attention, and track them.")
    parser.add_argument(
        '--data_dir', type=str, default='/nfs/tier3/projects/Cell_Model/data/processed/processed_visem/',
        help="Path to the root data directory containing class folders."
    )
    parser.add_argument(
        '--checkpoint_path', type=str, required=False,
        default='/nfs/tier3/projects/Cell_Model/experiments/v1/outputs/siamese_foundational/vit_b_16/v1_sperm_dataset/vit_b_16_Adam_lr_5e-06_bs_32_e_100/best.pth',
        help="Path to the trained model checkpoint (.pth file)."
    )
    parser.add_argument(
        '-b', '--backbone', type=str, default="vit_b_16",
        help="Network backbone used (must match checkpoint)."
    )
    parser.add_argument(
        '--img_size', type=int, default=224,
        help="Image size used during training."
    )
    parser.add_argument(
        '--attn_threshold_quantile', type=float, default=0.7,
        help="Quantile threshold for attention map binarization."
    )
    parser.add_argument(
        '--min_cell_area', type=int, default=20,
        help="Minimum pixel area for a detected cell region."
    )
    parser.add_argument(
        '--tracking_threshold', type=float, default=0.6,
        help="Minimum cosine similarity for Hungarian matching track."
    )
    parser.add_argument(
        '--max_pairs', type=int, default=10,
        help="Maximum number of pairs to visualize (default: 10)."
    )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- Data Prep ---
    class_image_data = scan_and_prepare_data(args.data_dir)
    if not class_image_data: sys.exit(1)
    image_pairs_with_labels = generate_pairs(class_image_data)
    if not image_pairs_with_labels:
        logger.info("No image pairs generated. Exiting.")
        sys.exit(1)
    image_pairs_with_labels = image_pairs_with_labels[:args.max_pairs] # Limit pairs
    logger.info(f"Processing {len(image_pairs_with_labels)} pairs.")


    # --- Transforms ---
    img_transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Load Model ---
    logger.info(f"Loading model with backbone: {args.backbone}")
    try:
        # Pass return_attention and return_patch_tokens flags to the V2 SiameseNetwork constructor
        model = SiameseNetwork(backbone=args.backbone, return_attention=True, return_patch_tokens=True)
        logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if 'model_state_dict' not in checkpoint:
            raise KeyError("Checkpoint does not contain 'model_state_dict'.")
        # Load state dict, handle potential DataParallel prefix
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")
        patch_size = model.backbone.patch_size
        logger.info(f"Model patch size: {patch_size}")
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found at: {args.checkpoint_path}"); sys.exit(1)
    except AttributeError:
         logger.error("Could not determine patch_size from model.backbone."); sys.exit(1)
    except KeyError as e:
        logger.error(f"Error loading checkpoint: Missing key {e}"); sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading checkpoint or model: {e}"); sys.exit(1)


    # --- Inference, Localization, Tracking, Visualization Loop ---
    batch_idx = 0 # Process one pair at a time

    for i, (path1, path2, true_label) in enumerate(image_pairs_with_labels):
        pair_type = "Positive" if true_label == 1 else "Negative"
        logger.info(f"\n--- Processing {pair_type} Pair {i+1}/{len(image_pairs_with_labels)} ---")
        logger.info(f"Img1: {os.path.basename(path1)}")
        logger.info(f"Img2: {os.path.basename(path2)}")

        # Initialize visualization components
        raw1_bgr, vis1_overlay = None, None
        raw2_bgr, vis2_overlay = None, None
        boxes1, centroids1, mask1 = [], [], None
        boxes2, centroids2, mask2 = [], [], None
        attn_grid1, attn_grid2 = None, None
        patch_tokens1, patch_tokens2 = None, None
        cell_embeddings1, cell_embeddings2 = [], []
        matches = []
        pred_prob_text = "Pred: N/A"
        true_label_text = f"True: {pair_type}"

        try:
            # Load and transform images
            img1_pil = Image.open(path1).convert('RGB')
            img2_pil = Image.open(path2).convert('RGB')
            img1_tensor = img_transform(img1_pil).unsqueeze(0).to(device)
            img2_tensor = img_transform(img2_pil).unsqueeze(0).to(device)
        except FileNotFoundError as e: logger.error(f"Error loading image: {e}. Skipping pair."); continue
        except Exception as e: logger.error(f"Error processing images: {e}. Skipping pair."); continue

        # --- Inference ---
        try:
            with torch.no_grad():
                # Use the get_backbone_outputs method from V2 model
                output1 = model.get_backbone_outputs(img1_tensor)
                output2 = model.get_backbone_outputs(img2_tensor)

                # Unpack based on expected V2 backbone output (features, attn_list, patch_tokens)
                # Handle cases where attention/patch_tokens might be None if flags were False
                features1, attn_list1, patch_tokens1 = None, None, None
                if isinstance(output1, tuple) and len(output1) >= 3:
                    features1, attn_list1, patch_tokens1 = output1
                elif isinstance(output1, tuple) and len(output1) == 2:
                    features1, attn_list1 = output1 # Assume attn only, no patches
                elif torch.is_tensor(output1):
                    features1 = output1 # Assume only features returned
                else:
                    logger.error(f"Unexpected backbone output type for img1: {type(output1)}")
                    continue

                features2, attn_list2, patch_tokens2 = None, None, None
                if isinstance(output2, tuple) and len(output2) >= 3:
                    features2, attn_list2, patch_tokens2 = output2
                elif isinstance(output2, tuple) and len(output2) == 2:
                    features2, attn_list2 = output2
                elif torch.is_tensor(output2):
                    features2 = output2
                else:
                    logger.error(f"Unexpected backbone output type for img2: {type(output2)}")
                    continue

                # Manually calculate Similarity Prediction using CLS features
                if features1 is not None and features2 is not None:
                     if features1.dim() != 2 or features2.dim() != 2:
                         logger.warning(f"Expected CLS features to have 2 dims [B, C], got {features1.shape} and {features2.shape}. Cannot compute similarity.")
                     else:
                         combined_features = torch.cat([features1, features2], dim=-1)
                         logits = model.cls_head(combined_features)
                         probability = torch.sigmoid(logits)
                         pred_prob = probability[batch_idx].item()
                         pred_prob_text = f"Pred (Sim): {pred_prob:.4f}"
                         logger.info(pred_prob_text)
                else:
                     logger.warning("Could not get CLS features, cannot calculate prediction.")

                # Move patch tokens to CPU if they exist
                patch_tokens1 = patch_tokens1.cpu() if patch_tokens1 is not None else None
                patch_tokens2 = patch_tokens2.cpu() if patch_tokens2 is not None else None

                # Process Attention Maps (if available)
                if attn_list1 is not None and len(attn_list1) > 0 and attn_list1[-1] is not None and len(attn_list1[-1]) > batch_idx:
                    last_layer_attn1 = attn_list1[-1][batch_idx].cpu()
                    num_tokens1 = last_layer_attn1.shape[-1]
                    num_patches1 = num_tokens1 - 1
                    if num_patches1 > 0:
                        attn_cls_to_patches1 = last_layer_attn1[0, 1:]
                        H_attn = W_attn = int(math.sqrt(num_patches1))
                        if H_attn * W_attn == num_patches1:
                            attn_grid1 = attn_cls_to_patches1.reshape(H_attn, W_attn).numpy()
                        else: logger.warning(f"Cannot reshape patch attention (num_patches={num_patches1}) for img1.")
                else: logger.info("No valid attention maps returned for img1.")

                if attn_list2 is not None and len(attn_list2) > 0 and attn_list2[-1] is not None and len(attn_list2[-1]) > batch_idx:
                     last_layer_attn2 = attn_list2[-1][batch_idx].cpu()
                     num_tokens2 = last_layer_attn2.shape[-1]
                     num_patches2 = num_tokens2 - 1
                     if num_patches2 > 0:
                         attn_cls_to_patches2 = last_layer_attn2[0, 1:]
                         H_attn = W_attn = int(math.sqrt(num_patches2))
                         if H_attn * W_attn == num_patches2:
                             attn_grid2 = attn_cls_to_patches2.reshape(H_attn, W_attn).numpy()
                         else: logger.warning(f"Cannot reshape patch attention (num_patches={num_patches2}) for img2.")
                else: logger.info("No valid attention maps returned for img2.")

        except AttributeError as e:
             logger.error(f"Model missing expected attribute/method (check get_backbone_outputs): {e}", exc_info=True); continue
        except Exception as e:
             logger.error(f"Error during model inference: {e}", exc_info=True); continue

        # --- Localization (Step 4) ---
        H_grid, W_grid = 0, 0
        if attn_grid1 is not None:
            H_grid, W_grid = attn_grid1.shape
            logger.info("Localizing cells in Image 1...")
            boxes1, centroids1, mask1 = localize_cells_from_attention(
                attn_grid1, args.img_size, patch_size, args.attn_threshold_quantile, args.min_cell_area
            )
            logger.info(f"Found {len(boxes1)} potential cells in Image 1.")
        else:
             logger.info("Skipping localization for Image 1 (no attention grid).")

        if attn_grid2 is not None:
            if H_grid == 0: H_grid, W_grid = attn_grid2.shape
            logger.info("Localizing cells in Image 2...")
            boxes2, centroids2, mask2 = localize_cells_from_attention(
                attn_grid2, args.img_size, patch_size, args.attn_threshold_quantile, args.min_cell_area
            )
            logger.info(f"Found {len(boxes2)} potential cells in Image 2.")
        else:
            logger.info("Skipping localization for Image 2 (no attention grid).")


        # --- Tracking (Step 5 - Only for Positive Pairs) ---
        if true_label == 1 and boxes1 and boxes2 and patch_tokens1 is not None and patch_tokens2 is not None:
            # Ensure grid dimensions were set correctly
            if H_grid == 0 or W_grid == 0:
                logger.warning("Grid dimensions H_grid/W_grid not set, cannot extract embeddings. Trying default 14x14.")
                H_grid, W_grid = (14, 14) # Default for 224 image, 16 patch

            logger.info("Extracting cell embeddings for tracking...")
            cell_embeddings1 = extract_cell_embeddings(patch_tokens1, centroids1, patch_size, H_grid, W_grid)
            cell_embeddings2 = extract_cell_embeddings(patch_tokens2, centroids2, patch_size, H_grid, W_grid)

            if cell_embeddings1 and cell_embeddings2:
                logger.info("Matching cells between frames...")
                matches = match_cells_hungarian(cell_embeddings1, cell_embeddings2, args.tracking_threshold)
                logger.info(f"Found {len(matches)} tracks.")
            else:
                logger.warning("Could not extract embeddings for tracking (check H_grid/W_grid and patch_tokens).")
        elif true_label == 1:
             logger.info("Skipping tracking (no cells detected in one or both images, or patch tokens unavailable).")


        # --- Visualization ---
        try:
            # Create base visualizations (raw BGR and attention overlay)
            if attn_grid1 is not None:
                 raw1_bgr, vis1_overlay = create_single_visualization(img1_tensor[batch_idx], torch.from_numpy(attn_grid1), patch_size, args.img_size)
            else:
                 raw1_bgr, _ = create_single_visualization(img1_tensor[batch_idx], torch.zeros(1,1), patch_size, args.img_size)

            if attn_grid2 is not None:
                 raw2_bgr, vis2_overlay = create_single_visualization(img2_tensor[batch_idx], torch.from_numpy(attn_grid2), patch_size, args.img_size)
            else:
                 raw2_bgr, _ = create_single_visualization(img2_tensor[batch_idx], torch.zeros(1,1), patch_size, args.img_size)

            # Draw bounding boxes on raw images
            box_color = (0, 255, 0)
            thickness = 1
            if raw1_bgr is not None and boxes1:
                for (x, y, w, h) in boxes1:
                    cv2.rectangle(raw1_bgr, (x, y), (x + w, y + h), box_color, thickness)
            if raw2_bgr is not None and boxes2:
                for (x, y, w, h) in boxes2:
                    cv2.rectangle(raw2_bgr, (x, y), (x + w, y + h), box_color, thickness)

            # Draw tracking markers (blue circles) on matched centroids
            track_color = (255, 0, 0)
            if true_label == 1 and matches:
                for idx1, idx2 in matches:
                    if idx1 < len(centroids1) and raw1_bgr is not None:
                        cv2.circle(raw1_bgr, centroids1[idx1], 3, track_color, -1)
                    if idx2 < len(centroids2) and raw2_bgr is not None:
                        cv2.circle(raw2_bgr, centroids2[idx2], 3, track_color, -1)

            # Create a dedicated tracking visualization with colored lines
            tracking_vis = None
            if true_label == 1 and matches and raw1_bgr is not None and raw2_bgr is not None:
                # Create a side-by-side visualization
                tracking_vis = np.zeros((args.img_size, args.img_size*2 + 20, 3), dtype=np.uint8)
                # Copy the two images with a gap between them
                tracking_vis[:, :args.img_size] = raw1_bgr.copy()
                tracking_vis[:, args.img_size+20:] = raw2_bgr.copy()
                
                # Draw a white separator
                tracking_vis[:, args.img_size:args.img_size+20] = (255, 255, 255)
                
                # Define a colormap for tracks (8 distinct colors)
                colors = [
                    (0, 0, 255),    # Red
                    (0, 255, 0),    # Green
                    (255, 0, 0),    # Blue
                    (0, 255, 255),  # Yellow
                    (255, 0, 255),  # Magenta
                    (255, 255, 0),  # Cyan
                    (128, 0, 128),  # Purple
                    (0, 128, 128),  # Teal
                ]
                
                # Draw tracking lines with unique colors and ID numbers
                for idx, (idx1, idx2) in enumerate(matches):
                    if idx1 < len(centroids1) and idx2 < len(centroids2):
                        color = colors[idx % len(colors)]
                        
                        # Draw larger circles at cell centroids
                        cv2.circle(tracking_vis, centroids1[idx1], 6, color, -1)
                        cv2.circle(tracking_vis, 
                                (centroids2[idx2][0] + args.img_size + 20, centroids2[idx2][1]), 
                                6, color, -1)
                        
                        # Draw connecting line across the images
                        cv2.line(tracking_vis, 
                                centroids1[idx1],
                                (centroids2[idx2][0] + args.img_size + 20, centroids2[idx2][1]),
                                color, 2)
                        
                        # Add numeric IDs by each cell
                        cv2.putText(tracking_vis, f"{idx+1}", 
                                   (centroids1[idx1][0] + 7, centroids1[idx1][1] + 7),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(tracking_vis, f"{idx+1}", 
                                   (centroids2[idx2][0] + args.img_size + 27, centroids2[idx2][1] + 7),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add title to tracking visualization
                title_text = f"Cell Tracking - {len(matches)} tracks"
                cv2.putText(tracking_vis, title_text, 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (255, 255, 255), 2)

            # Prepare final display panels
            panel1 = raw1_bgr if raw1_bgr is not None else np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
            panel2 = vis1_overlay if vis1_overlay is not None else np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
            panel3 = raw2_bgr if raw2_bgr is not None else np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
            panel4 = vis2_overlay if vis2_overlay is not None else np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)

            # --- Create the 4-panel visualization ---
            # Top row: Raw Img1 (with boxes/tracks), Attn Img1
            # Bottom row: Raw Img2 (with boxes/tracks), Attn Img2
            top_row = np.hstack((panel1, panel2))
            bottom_row = np.hstack((panel3, panel4))
            combined_vis = np.vstack((top_row, bottom_row))

            # Add Text Overlays (Prediction, True Label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (255, 255, 255); bg_color = (0, 0, 0); line_type = 1
            text_y_pred = 30; text_y_true = 55; text_x = 10

            (pred_w, pred_h), pred_b = cv2.getTextSize(pred_prob_text, font, font_scale, line_type)
            cv2.rectangle(combined_vis, (text_x, text_y_pred - pred_h - pred_b), (text_x + pred_w, text_y_pred + pred_b//2), bg_color, -1)
            cv2.putText(combined_vis, pred_prob_text, (text_x, text_y_pred), font, font_scale, font_color, line_type)

            (true_w, true_h), true_b = cv2.getTextSize(true_label_text, font, font_scale, line_type)
            cv2.rectangle(combined_vis, (text_x, text_y_true - true_h - true_b), (text_x + true_w, text_y_true + true_b//2), bg_color, -1)
            cv2.putText(combined_vis, true_label_text, (text_x, text_y_true), font, font_scale, font_color, line_type)

            # Add Cell Count Text
            count_text = f"Cells: {len(boxes1)} | {len(boxes2)}"
            (count_w, count_h), count_b = cv2.getTextSize(count_text, font, font_scale, line_type)
            text_y_count = text_y_true + 25
            cv2.rectangle(combined_vis, (text_x, text_y_count - count_h - count_b), (text_x + count_w, text_y_count + count_b//2), bg_color, -1)
            cv2.putText(combined_vis, count_text, (text_x, text_y_count), font, font_scale, font_color, line_type)
            
            # Create output directory for saving visualizations
            output_dir = os.path.join(os.path.dirname(args.checkpoint_path), "tracking_visualizations")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the visualizations to disk
            # Generate descriptive filenames
            base_filename = f"pair_{i+1}_{pair_type}"
            panels_path = os.path.join(output_dir, f"{base_filename}_panels.jpg")
            cv2.imwrite(panels_path, combined_vis)
            logger.info(f"Saved panels visualization to: {panels_path}")
            
            if tracking_vis is not None:
                tracking_path = os.path.join(output_dir, f"{base_filename}_tracks.jpg")
                cv2.imwrite(tracking_path, tracking_vis)
                logger.info(f"Saved tracking visualization to: {tracking_path}")
            
            # Try to display images (will likely fail in SSH environment)
            try:
                window_title = f"Pair {i+1} ({pair_type})"
                cv2.imshow(window_title, combined_vis)
                
                if tracking_vis is not None:
                    track_window_title = f"Tracking {i+1} ({pair_type})"
                    cv2.imshow(track_window_title, tracking_vis)
                
                logger.info(f"Displaying visualization. Press any key to continue...")
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
                if key == ord('q'): # Allow quitting early
                     logger.info("Quit key pressed. Exiting visualization loop.")
                     break
            except Exception as display_error:
                logger.warning(f"Could not display images (likely running headless): {display_error}")
                logger.info("Images saved to disk instead.")

        except Exception as viz_error:
            logger.error(f"Error during visualization for pair {i+1}: {viz_error}", exc_info=True)
            cv2.destroyAllWindows() # Clean up if error occurs mid-display
            continue # Skip to next pair

    logger.info("\nLocalization and Tracking visualization finished.")
    cv2.destroyAllWindows() # Final cleanup 