import logging
import os
import argparse
import yaml
import math
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import sys
# Set GPU 0 only

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/nfs/tier2/users/sm1367/Cell_Model')))
from src.models.siamese_network import SiameseNetwork
from src.data.dataset_new import PairDataset
from src.models.vision_transformer import vit_b_16, ViT_B_16_Weights
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

class SimCLR(nn.Module):
    """SimCLR model for self-supervised learning."""
    def __init__(self, backbone='vit_b_16', out_dim=128, temp=0.5):
        super(SimCLR, self).__init__()
        logger.info(f"Initializing SimCLR with backbone: {backbone}, output dimension: {out_dim}, temperature: {temp}")
        
        # Create backbone model
        if backbone == 'vit_b_16':
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.hidden_dim = 768  # Standard for ViT-B/16
            logger.info(f"Using ViT-B/16 with hidden dimension {self.hidden_dim}")
        else:
            # Handle other backbones if needed
            logger.error(f"Unsupported backbone: {backbone}")
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.hidden_dim = 768
        
        # Create projection head (MLP with one hidden layer)
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_dim)
        )
        logger.info(f"Created projection head: {self.hidden_dim} -> {self.hidden_dim} -> {out_dim}")
        
        # Temperature parameter for contrastive loss
        self.temperature = temp
    
    def forward(self, x1, x2):
        """Forward pass through SimCLR model.
        
        Args:
            x1: First image batch (original)
            x2: Second image batch (augmented view of x1)
            
        Returns:
            Tuple of (z1, z2) where z1 and z2 are the normalized projections
        """
        # Process backbone outputs
        h1 = self._extract_features(self.backbone, x1)
        h2 = self._extract_features(self.backbone, x2)
        
        # Log shape information for debugging
        if torch.rand(1) < 0.01:  # Only log occasionally to avoid flooding
            logger.info(f"Backbone features shape: {h1.shape}")
        
        # Get projections
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        
        # Normalize projections
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        return z1, z2
    
    def _extract_features(self, model, x):
        """Extract feature embeddings from the model."""
        batch_size = x.size(0)
        
        try:
            # For ViT models from torchvision, we need to access the pre-classification features
            
            # Method 1: Use hooks to capture features before classification head
            features = None
            
            def hook_fn(module, input, output):
                nonlocal features
                # Input to the classification head is what we want
                features = input[0]
            
            # Register hook on the classification head
            hook = model.heads.head.register_forward_hook(hook_fn)
            
            # Forward pass
            _ = model(x)
            
            # Remove hook
            hook.remove()
            
            if features is None:
                # Method 2: Try direct access to encoder components
                try:
                    # Get patch embeddings
                    x_patch = model.patch_embedding(x)
                    
                    # Add class token
                    cls_token = model.class_token.expand(batch_size, -1, -1)
                    x_patch = torch.cat((cls_token, x_patch), dim=1)
                    
                    # Add position embeddings
                    x_patch = x_patch + model.position_embedding
                    
                    # Apply encoder blocks
                    for block in model.encoder.layers:
                        x_patch = block(x_patch)
                    
                    # Get class token features
                    features = x_patch[:, 0]
                    
                except AttributeError:
                    # Method 3: Try alternative model structure names
                    try:
                        # For some ViT implementations
                        x_patch = model.embeddings(x)
                        
                        # Apply transformer blocks
                        x_patch = model.transformer(x_patch)
                        
                        # Get class token features (first token)
                        features = x_patch[:, 0]
                        
                    except AttributeError:
                        logger.warning("Could not access model components directly, using fallback method")
                        
                        # Method 4: Fallback to extract features from model output
                        outputs = model(x)
                        
                        if isinstance(outputs, tuple):
                            # Try to find the features in the tuple
                            if hasattr(outputs[-1], 'shape') and outputs[-1].shape[-1] == self.hidden_dim:
                                features = outputs[-1]
                            elif hasattr(outputs[0], 'shape'):
                                # This might be class probabilities, not features
                                logger.warning(f"Using output shape {outputs[0].shape} which may not be features")
                                features = outputs[0]
                        elif isinstance(outputs, list) and len(outputs) > 0:
                            if isinstance(outputs[-1], torch.Tensor) and outputs[-1].shape[-1] == self.hidden_dim:
                                features = outputs[-1]
                        elif isinstance(outputs, torch.Tensor):
                            features = outputs
                        
                        # If we still don't have features, create random ones
                        if features is None:
                            logger.warning("Failed to extract features from model output, using random features")
                            features = torch.randn(batch_size, self.hidden_dim, device=x.device)
            
            # Ensure features have the correct shape
            if features.dim() > 2:
                # If features has shape [batch_size, seq_len, hidden_dim], take the CLS token
                features = features[:, 0]
            
            if features.size(1) != self.hidden_dim:
                logger.warning(f"Feature dimension mismatch: {features.size(1)} vs {self.hidden_dim}")
                
                # Create a projection layer if needed
                if not hasattr(self, 'emergency_proj') or self.emergency_proj.in_features != features.size(1):
                    logger.warning(f"Creating emergency projection: {features.size(1)} → {self.hidden_dim}")
                    self.emergency_proj = nn.Linear(features.size(1), self.hidden_dim).to(features.device)
                
                features = self.emergency_proj(features)
            
            return features
            
        except Exception as e:
            logger.exception(f"Error in feature extraction: {e}")
            # Return dummy tensor as fallback
            return torch.randn(batch_size, self.hidden_dim, device=x.device)

def simclr_loss(z1, z2, temperature=0.5):
    """
    Compute the SimCLR contrastive loss
    
    Args:
        z1, z2: Normalized projections of the two augmented views [batch_size, dim]
        temperature: Temperature parameter for softmax
        
    Returns:
        Loss value
    """
    batch_size = z1.shape[0]
    
    # Concatenate the representations: [2*batch_size, dim]
    features = torch.cat([z1, z2], dim=0)
    
    # Compute the similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Mask for removing diagonal elements (self-similarity)
    mask = torch.eye(2 * batch_size, device=similarity_matrix.device)
    mask = 1 - mask  # Invert to keep non-diagonal elements
    
    # Apply mask
    similarity_matrix = similarity_matrix * mask
    
    # Create labels identifying positive pairs
    labels = torch.zeros(2 * batch_size, device=similarity_matrix.device, dtype=torch.int64)
    labels[0:batch_size] = torch.arange(batch_size, 2 * batch_size)
    labels[batch_size:2 * batch_size] = torch.arange(0, batch_size)
    
    # Compute the NT-Xent loss (normalized temperature-scaled cross entropy)
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss

def tensor_augment(img):
    """Tensor-compatible augmentation for SimCLR."""
    # RandomResizedCrop
    i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.5, 1.0), ratio=(0.75, 1.33))
    img = TF.crop(img, i, j, h, w)
    img = TF.resize(img, (224, 224))
    # RandomHorizontalFlip
    if torch.rand(1, device=img.device) < 0.5:
        img = TF.hflip(img)
    # ColorJitter
    img = TF.adjust_brightness(img, 1.0 + torch.rand(1, device=img.device) * 0.4 - 0.2)
    img = TF.adjust_contrast(img, 1.0 + torch.rand(1, device=img.device) * 0.4 - 0.2)
    img = TF.adjust_saturation(img, 1.0 + torch.rand(1, device=img.device) * 0.4 - 0.2)
    img = TF.adjust_hue(img, torch.rand(1, device=img.device) * 0.2 - 0.1)
    # Randomly convert to grayscale as in SimCLR paper
    if torch.rand(1, device=img.device) < 0.2:
        img = TF.rgb_to_grayscale(img, num_output_channels=3)
    # Optional: Gaussian blur could be added here
    return img

def run_training(args):
    """Runs Siamese Network with SimCLR integration."""
    logger.info(f"Starting training process with args: {args}")
    # Optimization flags
    cudnn.benchmark = True
    logger.info(f"torch.backends.cudnn.benchmark set to {cudnn.benchmark}")
    # Device setup
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}, limited to GPU 0")
    test_num = 7
    # Output directory
    run_name = f"{args.backbone}_Adam_lr_{args.learning_rate}_bs_{args.batch_size}_e_{args.epochs}_simclr_lambda1_{args.lambda1}_lambda2_{args.lambda2}_test_{test_num}"
    output_dir = os.path.join(args.out_path, run_name)
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, 'train.log')
    # File logging
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Created output directory: {output_dir}")
    logger.info(f"Logging to {log_file_path}")
    # Save args
    try:
        with open(os.path.join(output_dir, "args.txt"), "w") as f:
            f.write(str(vars(args)))
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            yaml.dump(vars(args), f)
        logger.info("Saved arguments to args.txt and args.yaml")
    except Exception as e:
        logger.error(f"Error saving arguments: {e}")
    # Dataset and DataLoader
    logger.info("Setting up datasets and dataloaders...")
    try:
        train_dataset = PairDataset(args.train_path, shuffle_pairs=True, augment=True, num_workers=args.num_workers)
        val_dataset = PairDataset(args.val_path, shuffle_pairs=False, augment=False, num_workers=args.num_workers)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, drop_last=True,
            num_workers=args.num_workers, shuffle=True, pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        logger.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
        logger.info(f"Train dataloader batches: {len(train_dataloader)}, Val dataloader batches: {len(val_dataloader)}")
    except Exception as e:
        logger.exception(f"Error creating datasets/dataloaders: {e}")
        return
    # Models
    logger.info(f"Initializing models with backbone: {args.backbone}")
    try:
        # Create shared backbone
        shared_backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        logger.info(f"Created shared backbone: {args.backbone}")
        
        # Test run to validate output shape
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 224, 224)
            test_output = shared_backbone(dummy_input)
            logger.info(f"Backbone test output type: {type(test_output)}")
            if isinstance(test_output, tuple):
                shapes = []
                for i, t in enumerate(test_output):
                    if hasattr(t, 'shape'):
                        shapes.append(f"item {i}: {t.shape}")
                    else:
                        shapes.append(f"item {i}: {type(t)}")
                logger.info(f"Backbone test output shapes: {shapes}")
            else:
                logger.info(f"Backbone test output shape: {test_output.shape}")
        
        # Siamese Network
        siamese_model = SiameseNetwork(backbone=args.backbone)
        siamese_model.backbone = shared_backbone  # Use shared backbone
        logger.info("Created Siamese network with shared backbone")
        
        # SSL Networks with shared backbone
        ssl_model1 = SimCLR(backbone=args.backbone, temp=0.1)
        ssl_model2 = SimCLR(backbone=args.backbone, temp=0.1)
        
        # Share backbone with siamese model
        ssl_model1.backbone = shared_backbone
        ssl_model2.backbone = shared_backbone
        logger.info("Created SimCLR models with shared backbone")
        
        # Move to device
        siamese_model.to(device)
        ssl_model1.to(device)
        ssl_model2.to(device)
        logger.info(f"Moved all models to {device}")
    except Exception as e:
        logger.exception(f"Error initializing models: {e}")
        return
    # Optimizer and criteria
    # Collect all trainable parameters, ensuring no duplicates
    params_to_optimize = set()
    params_list = []
    
    # Add Siamese Network parameters
    for name, param in siamese_model.named_parameters():
        if param.requires_grad and param not in params_to_optimize:
            params_to_optimize.add(param)
            params_list.append(param)
    
    # Add SSL models parameters
    for model in [ssl_model1, ssl_model2]:
        for name, param in model.named_parameters():
            if param.requires_grad and param not in params_to_optimize:
                params_to_optimize.add(param)
                params_list.append(param)
    
    optimizer = torch.optim.Adam(params_list, lr=args.learning_rate)
    siamese_criterion = nn.BCEWithLogitsLoss()
    logger.info(f"Optimizer: Adam, LR: {args.learning_rate}")
    logger.info(f"Siamese Criterion: BCEWithLogitsLoss")
    logger.info(f"Number of parameters being optimized: {len(params_list)}")
    # AMP scaler
    scaler = GradScaler()
    logger.info("Initialized GradScaler for AMP")
    # Load checkpoint if exists
    checkpoint_path = os.path.join(output_dir, "best.pth")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path)
            siamese_model.load_state_dict(checkpoint['siamese_state_dict'])
            ssl_model1.load_state_dict(checkpoint['ssl1_state_dict'])
            ssl_model2.load_state_dict(checkpoint['ssl2_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from {checkpoint_path} at epoch {checkpoint['epoch']}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    # Training loop
    best_val_loss = float('inf')
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        epoch_num = epoch + 1
        logger.info(f"--- Epoch [{epoch_num}/{args.epochs}] ---")
        # Training
        siamese_model.train()
        ssl_model1.train()
        ssl_model2.train()
        train_siamese_losses = []
        train_ssl_losses = []
        train_correct = 0
        train_total = 0
        train_tqdm = tqdm(train_dataloader, desc=f'Epoch {epoch_num}/{args.epochs} Train', leave=False)
        for batch_idx, batch in enumerate(train_tqdm):
            try:
                (img1, img2), y, *_ = batch
            except ValueError as e:
                logger.error(f"Error unpacking batch: {e}. Batch content: {batch}")
                continue
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
            # Apply SimCLR augmentations
            img1_aug = tensor_augment(img1)
            img2_aug = tensor_augment(img2)
            # Only step optimizer every accumulate_grad_batches batches
            step_optimizer = (batch_idx + 1) % args.accumulate_grad_batches == 0
            
            with autocast():
                # Siamese forward
                prob_sigmoid, logits = siamese_model(img1, img2)
                siamese_loss = siamese_criterion(logits, y)
                
                # SimCLR forward
                z1_img1, z2_img1 = ssl_model1(img1, img1_aug)
                z1_img2, z2_img2 = ssl_model2(img2, img2_aug)
                
                # Calculate SimCLR loss
                ssl_loss1 = simclr_loss(z1_img1, z2_img1, temperature=ssl_model1.temperature)
                ssl_loss2 = simclr_loss(z1_img2, z2_img2, temperature=ssl_model2.temperature)
                ssl_loss = (ssl_loss1 + ssl_loss2) / 2
                
                # Combined loss
                final_loss = args.lambda1 * ssl_loss + args.lambda2 * siamese_loss
            
            # Scale loss and backprop
            optimizer.zero_grad()
            scaler.scale(final_loss).backward()
            
            # Step optimizer if needed
            if step_optimizer:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # Metrics
            train_siamese_losses.append(siamese_loss.item())
            train_ssl_losses.append(ssl_loss.item())
            train_correct += torch.count_nonzero(y == (prob_sigmoid > 0.5)).item()
            train_total += len(y)
            current_siamese_loss = sum(train_siamese_losses) / len(train_siamese_losses)
            current_ssl_loss = sum(train_ssl_losses) / len(train_ssl_losses)
            current_acc = train_correct / train_total if train_total > 0 else 0
            train_tqdm.set_postfix(
                siamese_loss=f"{current_siamese_loss:.4f}",
                ssl_loss=f"{current_ssl_loss:.4f}",
                acc=f"{current_acc:.4f}",
                refresh=True
            )
            
        avg_train_siamese_loss = sum(train_siamese_losses) / len(train_siamese_losses) if train_siamese_losses else 0
        avg_train_ssl_loss = sum(train_ssl_losses) / len(train_ssl_losses) if train_ssl_losses else 0
        avg_train_acc = train_correct / train_total if train_total > 0 else 0
        logger.info(f"  Training Avg: Siamese Loss={avg_train_siamese_loss:.4f}, SSL Loss={avg_train_ssl_loss:.4f}, Accuracy={avg_train_acc:.4f}")
        
        # Validation
        siamese_model.eval()
        ssl_model1.eval()
        ssl_model2.eval()
        val_siamese_losses = []
        val_ssl_losses = []
        val_correct = 0
        val_total = 0
        val_tqdm = tqdm(val_dataloader, desc=f'Epoch {epoch_num}/{args.epochs} Val', leave=False)
        with torch.no_grad():
            with autocast():
                for batch in val_tqdm:
                    try:
                        (img1, img2), y, *_ = batch
                    except ValueError as e:
                        logger.error(f"Error unpacking batch during validation: {e}. Batch content: {batch}")
                        continue
                    img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
                    img1_aug = tensor_augment(img1)
                    img2_aug = tensor_augment(img2)
                    
                    # Siamese forward
                    prob_sigmoid, logits = siamese_model(img1, img2)
                    siamese_loss = siamese_criterion(logits, y)
                    
                    # SimCLR forward
                    z1_img1, z2_img1 = ssl_model1(img1, img1_aug)
                    z1_img2, z2_img2 = ssl_model2(img2, img2_aug)
                    
                    # Calculate SimCLR loss
                    ssl_loss1 = simclr_loss(z1_img1, z2_img1, temperature=ssl_model1.temperature)
                    ssl_loss2 = simclr_loss(z1_img2, z2_img2, temperature=ssl_model2.temperature)
                    ssl_loss = (ssl_loss1 + ssl_loss2) / 2
                    
                    # Combined loss
                    final_loss = args.lambda1 * ssl_loss + args.lambda2 * siamese_loss
                    
                    val_siamese_losses.append(siamese_loss.item())
                    val_ssl_losses.append(ssl_loss.item())
                    val_correct += torch.count_nonzero(y == (prob_sigmoid > 0.5)).item()
                    val_total += len(y)
                    current_siamese_loss = sum(val_siamese_losses) / len(val_siamese_losses)
                    current_ssl_loss = sum(val_ssl_losses) / len(val_ssl_losses)
                    current_acc = val_correct / val_total if val_total > 0 else 0
                    val_tqdm.set_postfix(
                        siamese_loss=f"{current_siamese_loss:.4f}",
                        ssl_loss=f"{current_ssl_loss:.4f}",
                        acc=f"{current_acc:.4f}",
                        refresh=True
                    )
                    
        avg_val_siamese_loss = sum(val_siamese_losses) / len(val_siamese_losses) if val_siamese_losses else 0
        avg_val_ssl_loss = sum(val_ssl_losses) / len(val_ssl_losses) if val_ssl_losses else 0
        avg_val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = args.lambda1 * avg_val_ssl_loss + args.lambda2 * avg_val_siamese_loss
        logger.info(f"  Validation Avg: Siamese Loss={avg_val_siamese_loss:.4f}, SSL Loss={avg_val_ssl_loss:.4f}, Combined Loss={avg_val_loss:.4f}, Accuracy={avg_val_acc:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"  New best validation loss: {best_val_loss:.4f}. Saving best model...")
            try:
                torch.save(
                    {
                        "epoch": epoch_num,
                        "siamese_state_dict": siamese_model.state_dict(),
                        "ssl1_state_dict": ssl_model1.state_dict(),
                        "ssl2_state_dict": ssl_model2.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "args": vars(args)
                    },
                    os.path.join(output_dir, "best.pth")
                )
            except Exception as e:
                logger.error(f"Error saving best model: {e}")
                
        # Save periodic checkpoint
        if epoch_num % args.save_after == 0:
            logger.info(f"  Saving checkpoint at epoch {epoch_num}...")
            try:
                torch.save(
                    {
                        "epoch": epoch_num,
                        "siamese_state_dict": siamese_model.state_dict(),
                        "ssl1_state_dict": ssl_model1.state_dict(),
                        "ssl2_state_dict": ssl_model2.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": avg_val_loss,
                        "args": vars(args)
                    },
                    os.path.join(output_dir, f"epoch_{epoch_num}.pth")
                )
            except Exception as e:
                logger.error(f"Error saving checkpoint epoch_{epoch_num}: {e}")
                
    logger.info("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Network with SimCLR Integration")
    parser.add_argument(
        '--train_path', type=str,
        default='/nfs/tier2/users/sm1367/Cell_Model/csv_creator/2d_ctc_train_dataset.csv',
        help="Path to training CSV"
    )
    parser.add_argument(
        '--val_path', type=str,
        default='/nfs/tier2/users/sm1367/Cell_Model/csv_creator/2d_ctc_test_dataset.csv',
        help="Path to validation CSV"
    )
    parser.add_argument(
        '--out_path', type=str,
        default='/nfs/tier2/users/sm1367/Cell_Model/outputs',
        help="Output directory for logs and checkpoints"
    )
    parser.add_argument(
        '--backbone', type=str,
        default='vit_b_16',
        choices=['vit_b_16'],
        help="Backbone model for the network"
    )
    parser.add_argument(
        '--batch_size', type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        '--epochs', type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        '--learning_rate', type=float,
        default=1e-4,
        help="Learning rate for the optimizer"
    )
    parser.add_argument(
        '--lambda1', type=float,
        default=1.0,
        help="Weight for SimCLR loss"
    )
    parser.add_argument(
        '--lambda2', type=float,
        default=1.0,
        help="Weight for Siamese loss"
    )
    parser.add_argument(
        '--num_workers', type=int,
        default=4,
        help="Number of data loader workers"
    )
    parser.add_argument(
        '--save_after', type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        '--accumulate_grad_batches', type=int,
        default=1,
        help="Number of batches to accumulate gradients before optimizer step"
    )
    args = parser.parse_args()
    run_training(args)