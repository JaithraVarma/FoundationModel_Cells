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
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/nfs/tier2/users/sm1367/Cell_Model')))
from src.models.siamese_network import SiameseNetwork
from src.data.dataset_new import PairDataset
from src.models.vision_transformer import vit_b_16, ViT_B_16_Weights

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')


class MultiViewDINO(nn.Module):
    """Enhanced DINO model with multi-view capabilities."""
    def __init__(self, backbone='vit_b_16', out_dim=2048, use_multi_crop=True):
        super(MultiViewDINO, self).__init__()
        logger.info(f"Initializing Multi-View DINO with backbone: {backbone}, output dimension: {out_dim}")
        
        # Create separate student and teacher models (same as your original)
        if backbone == 'vit_b_16':
            self.student = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.teacher = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.hidden_dim = 768  # Standard for ViT-B/16
        else:
            # Default fallback
            self.student = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.teacher = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.hidden_dim = 768
        
        # Create projection heads
        self.student_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, out_dim)
        )
        
        self.teacher_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, out_dim)
        )
        
        # Initialize teacher with student weights and freeze
        self._init_teacher()
        
        # Setup the center for teacher outputs
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.center_momentum = 0.9
        
        # Define multi-crop option
        self.use_multi_crop = use_multi_crop
    
    def _init_teacher(self):
        """Initialize teacher parameters with student parameters and freeze them."""
        logger.info("Initializing teacher with student weights and freezing parameters")
        # Copy weights from student to teacher
        for p_t, p_s in zip(self.teacher.parameters(), self.student.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False
        
        for p_t, p_s in zip(self.teacher_head.parameters(), self.student_head.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False
            
    def update_center(self, teacher_output):
        """Update center for teacher outputs using momentum."""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    def forward_teacher(self, x):
        """Forward pass through teacher network."""
        with torch.no_grad():
            feat = self._extract_features(self.teacher, x)
            proj = self.teacher_head(feat)
            return proj
            
    def forward_student(self, x):
        """Forward pass through student network."""
        feat = self._extract_features(self.student, x)
        proj = self.student_head(feat)
        return proj
    
    #only global
    def forward(self, x, views=None):
        """Forward pass with multiple global views.
        
        Args:
            x: Original image batch
            views: Optional pre-computed views
                
        Returns:
            Dictionary with student and teacher outputs
        """
        if views is None:
            # Generate views if not provided
            if self.use_multi_crop:
                global_views, local_views = multi_crop_transform(x)
                views = global_views + local_views
            else:
                # Just create two global views with different augmentations
                views = [
                    strong_augment(x),
                    strong_augment(x)
                ]
        
        # For the multi-global approach, we'll use the first two views as teacher views
        n_global = 2
        
        # Process all views through student
        student_outputs = []
        for view in views:
            student_outputs.append(self.forward_student(view))
        
        # Only process the first two views through teacher
        teacher_outputs = [self.forward_teacher(views[i]) for i in range(min(n_global, len(views)))]
        
        # Center the teacher outputs
        teacher_outputs = [t - self.center for t in teacher_outputs]
        
        # Update center with the teacher outputs
        with torch.no_grad():
            self.update_center(torch.cat(teacher_outputs))
        
        return {
            'student_outputs': student_outputs,
            'teacher_outputs': teacher_outputs,
            'n_global': n_global
        }

    #global and local
    # def forward(self, x, views=None):
    #     """Forward pass with multiple views.
        
    #     Args:
    #         x: Original image batch
    #         views: Optional pre-computed views
                
    #     Returns:
    #         Dictionary with student and teacher outputs
    #     """
    #     if views is None:
    #         # Generate views if not provided
    #         if self.use_multi_crop:
    #             global_views, local_views = multi_crop_transform(x)
    #             views = global_views + local_views
    #         else:
    #             # Just create two global views with different augmentations
    #             views = [
    #                 strong_augment(x),
    #                 strong_augment(x)
    #             ]
        
    #     # Number of global views (teacher views) is 2 if using multi-crop
    #     n_global = 2
        
    #     # Process all views through student
    #     student_outputs = []
    #     for i, view in enumerate(views):
    #         # Resize local views to 224x224 if they're smaller
    #         if i >= n_global and (view.shape[-2] != 224 or view.shape[-1] != 224):
    #             view = TF.resize(view, (224, 224))
    #         student_outputs.append(self.forward_student(view))
        
    #     # Only process global views through teacher
    #     teacher_outputs = [self.forward_teacher(views[i]) for i in range(min(n_global, len(views)))]
        
    #     # Center the teacher outputs
    #     teacher_outputs = [t - self.center for t in teacher_outputs]
        
    #     # Update center with the teacher outputs
    #     with torch.no_grad():
    #         self.update_center(torch.cat(teacher_outputs))
        
    #     return {
    #         'student_outputs': student_outputs,
    #         'teacher_outputs': teacher_outputs,
    #         'n_global': n_global
    #     }
    
    def _extract_features(self, model, x):
        """Extract feature embeddings from the model."""
        # Your original implementation
        batch_size = x.size(0)
        
        # Use hooks to capture features before classification head
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
        
        # Handle shape
        if features.dim() > 2:
            # If features has shape [batch_size, seq_len, hidden_dim], take the CLS token
            features = features[:, 0]
        
        return features

    def update_teacher(self, momentum=0.996):
        """Update teacher parameters using EMA."""
        with torch.no_grad():
            for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
                p_t.data = momentum * p_t.data + (1 - momentum) * p_s.data
            for p_s, p_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                p_t.data = momentum * p_t.data + (1 - momentum) * p_s.data


# Advanced augmentation functions
def strong_augment(img):
    """Strong augmentation pipeline for DINO."""
    # Get spatial dimensions
    device = img.device
    _, h, w = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[2], img.shape[3])
    
    # RandomResizedCrop with stronger scale variation
    i, j, h_crop, w_crop = transforms.RandomResizedCrop.get_params(
        img, scale=(0.08, 1.0), ratio=(0.75, 1.33))
    img = TF.crop(img, i, j, h_crop, w_crop)
    img = TF.resize(img, (224, 224))
    
    # Random horizontal flip
    if torch.rand(1) < 0.5:
        img = TF.hflip(img)
    
    # Color jitter with stronger parameters
    jitter_factor = 0.8
    img = TF.adjust_brightness(img, torch.rand(1, device=device) * 0.4 + 0.6)
    img = TF.adjust_contrast(img, torch.rand(1, device=device) * 0.4 + 0.6)
    img = TF.adjust_saturation(img, torch.rand(1, device=device) * 0.4 + 0.6)
    img = TF.adjust_hue(img, torch.rand(1, device=device) * 0.2 - 0.1)
    
    # Random grayscale
    if torch.rand(1) < 0.2:
        img = TF.rgb_to_grayscale(img, num_output_channels=3)
    
    # Gaussian blur (simulated with simple blur)
    if torch.rand(1) < 0.5:
        kernel_size = int(torch.randint(3, 7, (1,)).item())
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        sigma = torch.rand(1) * 1.9 + 0.1
        img = TF.gaussian_blur(img, kernel_size, sigma.item())
    
    # Solarization
    if torch.rand(1) < 0.2:
        img = torch.where(img >= 0.5, 1.0 - img, img)
    
    return img

#only global view
def multi_crop_transform(img):
    """Generate multiple views with only global crops."""
    device = img.device
    
    # Define number of global views to generate
    num_global_views = 8  # Instead of 2 global + 6 local
    
    global_views = []
    for _ in range(num_global_views):
        # Apply strong augmentation to create diverse global views
        global_view = strong_augment(img)
        global_views.append(global_view)
    
    # Return all views as global views, empty list for local views
    return global_views, []

#global view and local view
# def multi_crop_transform(img):
#     """Generate multiple views with global and local crops."""
#     device = img.device
    
#     # Global crops (224x224)
#     global_view1 = strong_augment(img)
#     global_view2 = strong_augment(img)
    
#     # Local crops (96x96)
#     local_views = []
#     for _ in range(6):  # Generate 6 local views
#         # Apply random crop with smaller size
#         i, j, h, w = transforms.RandomResizedCrop.get_params(
#             img, scale=(0.05, 0.3), ratio=(0.75, 1.33))
#         local_view = TF.crop(img, i, j, h, w)
#         local_view = TF.resize(local_view, (96, 96))

#         # Apply other augmentations - with device-aware random tensors
#         if torch.rand(1, device=device) < 0.5:
#             local_view = TF.hflip(local_view)
        
#         # Color jitter with device-aware random tensors
#         jitter_factor = 0.8
#         local_view = TF.adjust_brightness(local_view, 1.0 + torch.rand(1, device=device) * jitter_factor - jitter_factor/2)
#         local_view = TF.adjust_contrast(local_view, 1.0 + torch.rand(1, device=device) * jitter_factor - jitter_factor/2)
#         local_view = TF.adjust_saturation(local_view, 1.0 + torch.rand(1, device=device) * jitter_factor - jitter_factor/2)
#         local_view = TF.adjust_hue(local_view, torch.rand(1, device=device) * 0.2 - 0.1)
        
#         local_views.append(local_view)
    
#     return [global_view1, global_view2], local_views


def dino_loss_multi_view(outputs, student_temp=0.1, teacher_temp=0.04, sinkhorn_iterations=3):
    """Enhanced DINO loss with multiple views support and optional Sinkhorn-Knopp.
    
    Args:
        outputs: Dictionary containing student_outputs and teacher_outputs
        student_temp: Temperature for student
        teacher_temp: Temperature for teacher
        sinkhorn_iterations: Number of Sinkhorn-Knopp iterations (0 to disable)
        
    Returns:
        Loss value
    """
    student_outputs = outputs['student_outputs']
    teacher_outputs = outputs['teacher_outputs']
    n_global = outputs['n_global']
    
    # Teacher outputs to probability distributions
    teacher_logits = [t / teacher_temp for t in teacher_outputs]
    
    # Apply Sinkhorn-Knopp if requested
    if sinkhorn_iterations > 0:
        teacher_probs = [sinkhorn(t, sinkhorn_iterations) for t in teacher_logits]
    else:
        teacher_probs = [F.softmax(t, dim=-1) for t in teacher_logits]
    
    # Cross-entropy between all combinations of student and teacher views
    total_loss = 0
    n_loss_terms = 0
    
    # Only supervised by teacher (global) views
    for i, teacher_out in enumerate(teacher_probs):
        # Teacher view i teaches all student views
        for j, student_out in enumerate(student_outputs):
            # Skip if student and teacher view are the same
            if i == j:
                continue
                
            # Cross-entropy loss: teacher predicts student
            loss = -torch.mean(
                torch.sum(
                    teacher_out.detach() * F.log_softmax(student_out / student_temp, dim=-1),
                    dim=-1
                )
            )
            
            total_loss += loss
            n_loss_terms += 1
    
    # Normalize
    total_loss /= n_loss_terms
    
    return total_loss


def sinkhorn(scores, n_iters=3, epsilon=0.05):
    """Sinkhorn-Knopp normalization for DINO."""
    # Get batch size
    batch_size = scores.size(0)
    
    # Initialize uniform distribution
    Q = torch.ones_like(scores) / scores.size(1)
    
    # Sinkhorn iterations
    with torch.no_grad():
        for _ in range(n_iters):
            # Normalize each row: each sample gets equal importance
            sum_Q = torch.sum(Q, dim=1, keepdim=True)
            Q = Q / sum_Q
            
            # Normalize each column: uniform weights for each cluster
            Q = Q * (scores.size(1) / torch.sum(Q, dim=0, keepdim=True))
    
    # Apply softmax with rescaled scores
    return torch.softmax(scores / epsilon, dim=1)


class CosineScheduleWithWarmup:
    """Cosine learning rate schedule with warmup."""
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr_factor=0.1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr_factor = min_lr_factor
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        """Update learning rate based on epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = epoch / self.warmup_epochs
        else:
            # Cosine decay
            factor = self.min_lr_factor + 0.5 * (1 - self.min_lr_factor) * (
                1 + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))
            )
        
        # Update learning rates
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * factor
            
        return factor
    
def run_training(args):
    """Runs Siamese Network with enhanced Multi-View DINO integration."""
    # Device setup and optimization flags
    cudnn.benchmark = True
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    # Output directory setup
    run_name = f"{args.backbone}_Adam_lr_{args.learning_rate}_bs_{args.batch_size}_e_{args.epochs}_dino_lambda1_{args.lambda1}_lambda2_{args.lambda2}_views_{args.num_views}_test_{args.test_num}"
    output_dir = os.path.join(args.out_path, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    # Create file handler for logging to file
    log_file = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Log configuration
    logger.info(f"Starting training process with args: {args}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using device: {device}")
    
    # Dataset and DataLoader
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
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Models
    logger.info(f"Initializing models with backbone: {args.backbone}")
    
    # Create shared backbone
    shared_backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    
    # Siamese Network
    siamese_model = SiameseNetwork(backbone=args.backbone)
    siamese_model.backbone = shared_backbone  # Use shared backbone
    
    # SSL Networks - using new MultiViewDINO
    ssl_model1 = MultiViewDINO(
        backbone=args.backbone, 
        out_dim=args.dino_dim,
        use_multi_crop=(args.num_views > 2)
    )
    ssl_model2 = MultiViewDINO(
        backbone=args.backbone,
        out_dim=args.dino_dim,
        use_multi_crop=(args.num_views > 2)
    )
    
    # Share student backbone with siamese model
    ssl_model1.student = shared_backbone
    ssl_model2.student = shared_backbone
    
    # Move to device
    siamese_model.to(device)
    ssl_model1.to(device)
    ssl_model2.to(device)
    
    # Optimizer and parameters setup
    # Collect all trainable parameters
    params_to_optimize = set()
    params_list = []
    
    # Add Siamese Network parameters
    for name, param in siamese_model.named_parameters():
        if param.requires_grad and param not in params_to_optimize:
            params_to_optimize.add(param)
            params_list.append(param)
    
    # Add SSL models parameters (only student_head for ssl_model1 and ssl_model2)
    for model in [ssl_model1, ssl_model2]:
        for name, param in model.named_parameters():
            if 'teacher' not in name and param.requires_grad and param not in params_to_optimize:
                params_to_optimize.add(param)
                params_list.append(param)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        params_list, 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"DINO lambda1: {args.lambda1}")
    logger.info(f"Siamese lambda2: {args.lambda2}")
    logger.info(f"Number of views: {args.num_views}")
    
    # Learning rate scheduler
    lr_scheduler = CosineScheduleWithWarmup(
        optimizer, 
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        min_lr_factor=1
    )
    
    # Loss criteria
    siamese_criterion = nn.BCEWithLogitsLoss()
    
    # AMP scaler
    scaler = GradScaler()
    
    # Loss normalization constant for DINO
    dino_norm_factor = math.log(args.dino_dim)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    logger.info(f"Starting training for {args.epochs} epochs")

    for epoch in range(args.epochs):
        epoch_num = epoch + 1
        logger.info(f"--- Epoch [{epoch_num}/{args.epochs}] ---")
        
        # Update learning rate
        lr_factor = lr_scheduler.step(epoch/10)
        logger.info(f"  Learning rate factor: {lr_factor:.6f}")
        
        # Training
        siamese_model.train()
        ssl_model1.train()
        ssl_model2.train()
        train_siamese_losses = []
        train_ssl_losses = []
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch_num}/{args.epochs} Train')):
            # Unpack batch
            try:
                (img1, img2), y, _, _ = batch
            except ValueError as e:
                logger.error(f"Error unpacking batch: {e}")
                continue

            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
            
            # Only step optimizer every accumulate_grad_batches
            step_optimizer = (batch_idx + 1) % args.accumulate_grad_batches == 0
            
            with autocast():
                # Siamese forward
                prob_sigmoid, logits = siamese_model(img1, img2)
                siamese_loss = siamese_criterion(logits, y)

                # SSL forward with multi-view
                outputs1 = ssl_model1(img1)
                outputs2 = ssl_model2(img2)
                
                # Calculate DINO loss
                ssl_loss1 = dino_loss_multi_view(
                    outputs1, 
                    student_temp=args.student_temp,
                    teacher_temp=args.teacher_temp,
                    sinkhorn_iterations=args.sinkhorn_iters
                )
                ssl_loss2 = dino_loss_multi_view(
                    outputs2,
                    student_temp=args.student_temp,
                    teacher_temp=args.teacher_temp,
                    sinkhorn_iterations=args.sinkhorn_iters
                )

                # Normalize DINO loss
                ssl_loss1_normalized = ssl_loss1 / dino_norm_factor
                ssl_loss2_normalized = ssl_loss2 / dino_norm_factor
                ssl_loss = (ssl_loss1_normalized + ssl_loss2_normalized) / 2

                # Combined loss with warmup for DINO
                # Gradually increase the weight of DINO loss during warmup
                if epoch < args.warmup_epochs:
                    lambda1_current = args.lambda1 * (epoch / args.warmup_epochs)
                else:
                    lambda1_current = args.lambda1
                
                final_loss = lambda1_current * ssl_loss + args.lambda2 * siamese_loss

            # Scale loss and backprop
            optimizer.zero_grad()
            scaler.scale(final_loss).backward()
            
            # Step optimizer if needed
            if step_optimizer:
                # Gradient clipping
                if args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params_list, args.clip_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update teacher networks after optimizer step
                ssl_model1.update_teacher(momentum=args.teacher_momentum)
                ssl_model2.update_teacher(momentum=args.teacher_momentum)

            # Metrics
            train_siamese_losses.append(siamese_loss.item())
            train_ssl_losses.append(ssl_loss.item())
            train_correct += torch.count_nonzero(y == (prob_sigmoid > 0.5)).item()
            train_total += len(y)
        
        # Validation
        siamese_model.eval()
        ssl_model1.eval()
        ssl_model2.eval()
        val_siamese_losses = []
        val_ssl_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f'Epoch {epoch_num}/{args.epochs} Val'):
                try:
                    (img1, img2), y, _, _ = batch
                except ValueError as e:
                    logger.error(f"Error unpacking validation batch: {e}")
                    continue
                    
                img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
                
                # Siamese forward
                prob_sigmoid, logits = siamese_model(img1, img2)
                siamese_loss = siamese_criterion(logits, y)
                
                # SSL forward with multi-view
                outputs1 = ssl_model1(img1)
                outputs2 = ssl_model2(img2)
                
                # Calculate DINO loss
                ssl_loss1 = dino_loss_multi_view(
                    outputs1,
                    student_temp=args.student_temp,
                    teacher_temp=args.teacher_temp,
                    sinkhorn_iterations=args.sinkhorn_iters
                )
                ssl_loss2 = dino_loss_multi_view(
                    outputs2,
                    student_temp=args.student_temp,
                    teacher_temp=args.teacher_temp,
                    sinkhorn_iterations=args.sinkhorn_iters
                )
                
                # Normalize DINO loss
                ssl_loss1_normalized = ssl_loss1 / dino_norm_factor
                ssl_loss2_normalized = ssl_loss2 / dino_norm_factor
                ssl_loss = (ssl_loss1_normalized + ssl_loss2_normalized) / 2
                
                # Metrics
                val_siamese_losses.append(siamese_loss.item())
                val_ssl_losses.append(ssl_loss.item())
                val_correct += torch.count_nonzero(y == (prob_sigmoid > 0.5)).item()
                val_total += len(y)
        
        # Log metrics
        avg_train_siamese_loss = sum(train_siamese_losses) / len(train_siamese_losses) if train_siamese_losses else 0
        avg_train_ssl_loss = sum(train_ssl_losses) / len(train_ssl_losses) if train_ssl_losses else 0
        avg_train_acc = train_correct / train_total if train_total > 0 else 0
        
        avg_val_siamese_loss = sum(val_siamese_losses) / len(val_siamese_losses) if val_siamese_losses else 0
        avg_val_ssl_loss = sum(val_ssl_losses) / len(val_ssl_losses) if val_ssl_losses else 0
        avg_val_acc = val_correct / val_total if val_total > 0 else 0
        
        logger.info(f"  Train: Siamese Loss: {avg_train_siamese_loss:.4f}, SSL Loss: {avg_train_ssl_loss:.4f}, Accuracy: {avg_train_acc:.4f}")
        logger.info(f"  Val: Siamese Loss: {avg_val_siamese_loss:.4f}, SSL Loss: {avg_val_ssl_loss:.4f}, Accuracy: {avg_val_acc:.4f}")
        
        # Save checkpoint
        if epoch_num % 1 == 0 or epoch_num == args.epochs:
            logger.info(f"  Saving checkpoint at epoch {epoch_num}...")
            try:
                torch.save(
                    {
                        "epoch": epoch_num,
                        "siamese_state_dict": siamese_model.state_dict(),
                        "ssl1_state_dict": ssl_model1.state_dict(),
                        "ssl2_state_dict": ssl_model2.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": avg_val_acc,
                        "args": vars(args)
                    },
                    os.path.join(output_dir, f"checkpoint_ep{epoch_num}.pth")
                )
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")

        # Save best model based on accuracy
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            logger.info(f"  New best validation accuracy: {best_val_acc:.4f}. Saving best model...")
            try:
                torch.save(
                    {
                        "epoch": epoch_num,
                        "siamese_state_dict": siamese_model.state_dict(),
                        "ssl1_state_dict": ssl_model1.state_dict(),
                        "ssl2_state_dict": ssl_model2.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_acc": best_val_acc,
                        "args": vars(args)
                    },
                    os.path.join(output_dir, "best_acc.pth")
                )
            except Exception as e:
                logger.error(f"Error saving best model: {e}")

    logger.info("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Network with Multi-View DINO Integration")
    # Original arguments
    parser.add_argument('--train_path', type=str, default='/nfs/tier2/users/sm1367/Cell_Model/csv_creator/2d_ctc_train_dataset.csv')
    parser.add_argument('--val_path', type=str, default='/nfs/tier2/users/sm1367/Cell_Model/csv_creator/2d_ctc_test_dataset.csv')
    parser.add_argument('--out_path', type=str, default='/nfs/tier2/users/sm1367/Cell_Model/outputs')
    parser.add_argument('--backbone', type=str, default='vit_b_16', choices=['vit_b_16'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_after', type=int, default=10)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    
    # New arguments for multi-view DINO
    parser.add_argument('--num_views', type=int, default=8, help="Number of views (2=global only, 8=global+local)")
    parser.add_argument('--dino_dim', type=int, default=2048, help="Output dimension of DINO projection")
    parser.add_argument('--student_temp', type=float, default=0.1, help="Temperature for student outputs")
    parser.add_argument('--teacher_temp', type=float, default=0.04, help="Temperature for teacher outputs")
    parser.add_argument('--teacher_momentum', type=float, default=0.996, help="EMA momentum for teacher update")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="Weight decay for optimizer")
    parser.add_argument('--warmup_epochs', type=int, default=10, help="Number of warmup epochs")
    parser.add_argument('--clip_grad_norm', type=float, default=3.0, help="Gradient clipping norm")
    parser.add_argument('--sinkhorn_iters', type=int, default=3, help="Sinkhorn-Knopp iterations (0 to disable)")
    parser.add_argument('--test_num', type=int, default=1, help="Test run number")

    args = parser.parse_args()
    run_training(args)