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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/nfs/tier2/users/sm1367/Cell_Model')))
from src.models.siamese_network import SiameseNetwork
from src.data.dataset_new import PairDataset
from src.models.vision_transformer import vit_b_16, ViT_B_16_Weights

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

class DINO(nn.Module):
    """DINO model for self-supervised learning."""
    def __init__(self, backbone='vit_b_16', out_dim=2048):
        super(DINO, self).__init__()
        logger.info(f"Initializing DINO with backbone: {backbone}, output dimension: {out_dim}")
        
        if backbone == 'vit_b_16':
            self.student = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.teacher = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.hidden_dim = 768
            logger.info(f"Using ViT-B/16 with hidden dimension {self.hidden_dim}")
        else:
            logger.error(f"Unsupported backbone: {backbone}")
            self.student = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.teacher = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.hidden_dim = 768
        
        self.student_head = nn.Linear(self.hidden_dim, out_dim)
        self.teacher_head = nn.Linear(self.hidden_dim, out_dim)
        logger.info(f"Created projection heads: {self.hidden_dim} -> {out_dim}")
        
        self._init_teacher()
    
    def _init_teacher(self):
        logger.info("Initializing teacher with student weights and freezing parameters")
        for p_t, p_s in zip(self.teacher.parameters(), self.student.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False
        for p_t, p_s in zip(self.teacher_head.parameters(), self.student_head.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False

    def forward(self, x_list):
        """Forward pass for multiple views.
        
        Args:
            x_list: List of [batch_size, C, H, W] tensors (multiple views)
        
        Returns:
            List of (student_out, teacher_out) tuples for each view
        """
        outputs = []
        for x in x_list:
            s_out = self._extract_features(self.student, x)
            if torch.rand(1) < 0.01:
                logger.info(f"Student features shape: {s_out.shape}")
            s = self.student_head(s_out)
            with torch.no_grad():
                t_out = self._extract_features(self.teacher, x)
                t = self.teacher_head(t_out)
            outputs.append((s, t))
        return outputs
    
    def _extract_features(self, model, x):
        batch_size = x.size(0)
        try:
            features = None
            def hook_fn(module, input, output):
                nonlocal features
                features = input[0]
            hook = model.heads.head.register_forward_hook(hook_fn)
            _ = model(x)
            hook.remove()
            if features is None:
                try:
                    x_patch = model.patch_embedding(x)
                    cls_token = model.class_token.expand(batch_size, -1, -1)
                    x_patch = torch.cat((cls_token, x_patch), dim=1)
                    x_patch = x_patch + model.position_embedding
                    for block in model.encoder.layers:
                        x_patch = block(x_patch)
                    features = x_patch[:, 0]
                except AttributeError:
                    try:
                        x_patch = model.embeddings(x)
                        x_patch = model.transformer(x_patch)
                        features = x_patch[:, 0]
                    except AttributeError:
                        logger.warning("Could not access model components directly, using fallback method")
                        outputs = model(x)
                        if isinstance(outputs, tuple):
                            if hasattr(outputs[-1], 'shape') and outputs[-1].shape[-1] == self.hidden_dim:
                                features = outputs[-1]
                            elif hasattr(outputs[0], 'shape'):
                                logger.warning(f"Using output shape {outputs[0].shape} which may not be features")
                                features = outputs[0]
                        elif isinstance(outputs, list) and len(outputs) > 0:
                            if isinstance(outputs[-1], torch.Tensor) and outputs[-1].shape[-1] == self.hidden_dim:
                                features = outputs[-1]
                        elif isinstance(outputs, torch.Tensor):
                            features = outputs
                        if features is None:
                            logger.warning("Failed to extract features from model output, using random features")
                            features = torch.randn(batch_size, self.hidden_dim, device=x.device)
            if features.dim() > 2:
                features = features[:, 0]
            if features.size(1) != self.hidden_dim:
                logger.warning(f"Feature dimension mismatch: {features.size(1)} vs {self.hidden_dim}")
                if not hasattr(self, 'emergency_proj') or self.emergency_proj.in_features != features.size(1):
                    logger.warning(f"Creating emergency projection: {features.size(1)} -> {self.hidden_dim}")
                    self.emergency_proj = nn.Linear(features.size(1), self.hidden_dim).to(features.device)
                features = self.emergency_proj(features)
            return features
        except Exception as e:
            logger.exception(f"Error in feature extraction: {e}")
            return torch.randn(batch_size, self.hidden_dim, device=x.device)

    def update_teacher(self, momentum=0.996, epoch=None, max_epochs=None):
        if epoch is not None and max_epochs is not None:
            momentum = 0.996 + (0.999 - 0.996) * (epoch / max_epochs)
        with torch.no_grad():
            for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
                p_t.data = momentum * p_t.data + (1 - momentum) * p_s.data
            for p_s, p_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                p_t.data = momentum * p_t.data + (1 - momentum) * p_s.data

def dino_loss(s_list, t_list, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
    """DINO loss across multiple view pairs."""
    losses = []
    for s, t in zip(s_list, t_list):
        t = F.softmax(t / teacher_temp, dim=-1)
        loss = -torch.mean(torch.sum(t.detach() * F.log_softmax(s / student_temp, dim=-1), dim=-1))
        losses.append(loss)
    return sum(losses) / len(losses)

def tensor_augment(img, config):
    """Generate multiple augmented views for an image."""
    num_views = config['training'].get('num_views', 2)
    views = []
    for _ in range(num_views):
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, scale=config['augmentation']['random_resized_crop']['scale'],
            ratio=config['augmentation']['random_resized_crop']['ratio']
        )
        aug_img = TF.crop(img, i, j, h, w)
        aug_img = TF.resize(aug_img, (224, 224))
        if torch.rand(1, device=img.device) < config['augmentation']['hflip_prob']:
            aug_img = TF.hflip(aug_img)
        aug_img = TF.adjust_brightness(aug_img, 1.0 + torch.rand(1, device=img.device) * config['augmentation']['brightness_jitter'] - config['augmentation']['brightness_jitter']/2)
        aug_img = TF.adjust_contrast(aug_img, 1.0 + torch.rand(1, device=img.device) * config['augmentation']['contrast_jitter'] - config['augmentation']['contrast_jitter']/2)
        aug_img = TF.adjust_saturation(aug_img, 1.0 + torch.rand(1, device=img.device) * config['augmentation']['saturation_jitter'] - config['augmentation']['saturation_jitter']/2)
        aug_img = TF.adjust_hue(aug_img, torch.rand(1, device=img.device) * config['augmentation']['hue_jitter'] - config['augmentation']['hue_jitter']/2)
        if config['augmentation']['gaussian_blur_prob'] > 0 and torch.rand(1, device=img.device) < config['augmentation']['gaussian_blur_prob']:
            aug_img = TF.gaussian_blur(aug_img, kernel_size=config['augmentation']['gaussian_blur']['kernel_size'], sigma=config['augmentation']['gaussian_blur']['sigma'])
        views.append(aug_img)
    return views

def run_training(config):
    logger.info(f"Starting training process with config: {config}")

    cudnn.benchmark = True
    logger.info(f"torch.backends.cudnn.benchmark set to {cudnn.benchmark}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}, limited to GPU 0")
    test_num = config.get('test_num', 1)

    run_name = f"{config['model']['backbone']}_Adam_lr_{config['training']['learning_rate']}_bs_{config['training']['batch_size']}_e_{config['training']['epochs']}_dino_lambda1_{config['loss']['lambda1']}_lambda2_{config['loss']['lambda2']}_views_{config['training'].get('num_views', 2)}_test_{test_num}"
    output_dir = os.path.join(config['data']['out_path'], run_name)
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, 'train.log')

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Created output directory: {output_dir}")
    logger.info(f"Logging to {log_file_path}")

    try:
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
        logger.info("Saved config to config.yaml")
    except Exception as e:
        logger.error(f"Error saving config: {e}")

    logger.info("Setting up datasets and dataloaders...")
    try:
        train_dataset = PairDataset(config['data']['train_path'], shuffle_pairs=True, augment=True, num_workers=config['training']['num_workers'])
        val_dataset = PairDataset(config['data']['val_path'], shuffle_pairs=False, augment=False, num_workers=config['training']['num_workers'])
        train_dataloader = DataLoader(
            train_dataset, batch_size=config['training']['batch_size'], drop_last=True,
            num_workers=config['training']['num_workers'], shuffle=True, pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=config['training']['batch_size'], shuffle=False,
            num_workers=config['training']['num_workers'], pin_memory=True
        )
        logger.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
        logger.info(f"Train dataloader batches: {len(train_dataloader)}, Val dataloader batches: {len(val_dataloader)}")
    except Exception as e:
        logger.exception(f"Error creating datasets/dataloaders: {e}")
        return

    logger.info(f"Initializing models with backbone: {config['model']['backbone']}")
    try:
        shared_backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)
        logger.info(f"Created shared backbone: {config['model']['backbone']}")
        
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 224, 224).to(device)
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
        
        siamese_model = SiameseNetwork(backbone=config['model']['backbone']).to(device)
        siamese_model.backbone = shared_backbone
        logger.info("Created Siamese network with shared backbone")
        
        ssl_model1 = DINO(backbone=config['model']['backbone'], out_dim=config['model']['out_dim']).to(device)
        ssl_model2 = DINO(backbone=config['model']['backbone'], out_dim=config['model']['out_dim']).to(device)
        ssl_model1.student = shared_backbone
        ssl_model2.student = shared_backbone
        logger.info("Created DINO models with shared student backbone")

    except Exception as e:
        logger.exception(f"Error initializing models: {e}")
        return

    params_to_optimize = set()
    params_list = []
    for name, param in siamese_model.named_parameters():
        if param.requires_grad and param not in params_to_optimize:
            params_to_optimize.add(param)
            params_list.append(param)
    for model in [ssl_model1, ssl_model2]:
        for name, param in model.named_parameters():
            if 'teacher' not in name and param.requires_grad and param not in params_to_optimize:
                params_to_optimize.add(param)
                params_list.append(param)
    
    optimizer = torch.optim.Adam(params_list, lr=config['training']['learning_rate'])
    if config['training']['use_cosine_scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    else:
        scheduler = None
    siamese_criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    dino_norm_factor = config['loss']['dino_norm_factor']
    
    checkpoint_path = os.path.join(output_dir, "best.pth")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            siamese_model.load_state_dict(checkpoint['siamese_state_dict'])
            ssl_model1.load_state_dict(checkpoint['ssl1_state_dict'])
            ssl_model2.load_state_dict(checkpoint['ssl2_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Loaded checkpoint from {checkpoint_path} at epoch {checkpoint['epoch']}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")

    best_val_loss = float('inf')
    logger.info(f"Starting training for {config['training']['epochs']} epochs")

    for epoch in range(config['training']['epochs']):
        epoch_num = epoch + 1
        logger.info(f"--- Epoch [{epoch_num}/{config['training']['epochs']}] ---")

        siamese_model.train()
        ssl_model1.train()
        ssl_model2.train()
        train_siamese_losses = []
        train_ssl_losses = []
        train_ssl_raw_losses = []
        train_correct = 0
        train_total = 0
        train_tqdm = tqdm(train_dataloader, desc=f'Epoch {epoch_num}/{config["training"]["epochs"]} Train', leave=False)

        for batch_idx, batch in enumerate(train_tqdm):
            try:
                (img1, img2), y, _, _ = batch
            except ValueError as e:
                logger.error(f"Error unpacking batch: {e}. Batch content: {batch}")
                continue

            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
            img1_views = tensor_augment(img1, config)
            img2_views = tensor_augment(img2, config)

            step_optimizer = (batch_idx + 1) % config['training']['accumulate_grad_batches'] == 0
            
            with autocast():
                prob_sigmoid, logits = siamese_model(img1, img2)
                siamese_loss = siamese_criterion(logits, y)

                outputs1 = ssl_model1(img1_views)
                outputs2 = ssl_model2(img2_views)
                s1_list = [s for s, _ in outputs1]
                t1_list = [t for _, t in outputs1]
                s2_list = [s for s, _ in outputs2]
                t2_list = [t for _, t in outputs2]
                
                ssl_loss1 = dino_loss(s1_list, t1_list, config['loss']['student_temp'], config['loss']['teacher_temp'], config['loss']['center_momentum'])
                ssl_loss2 = dino_loss(s2_list, t2_list, config['loss']['student_temp'], config['loss']['teacher_temp'], config['loss']['center_momentum'])

                ssl_loss1_normalized = ssl_loss1 / dino_norm_factor
                ssl_loss2_normalized = ssl_loss2 / dino_norm_factor
                ssl_loss = (ssl_loss1_normalized + ssl_loss2_normalized) / 2

                final_loss = config['loss']['lambda1'] * ssl_loss + config['loss']['lambda2'] * siamese_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(final_loss).backward()
            
            if step_optimizer:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ssl_model1.update_teacher(config['training']['momentum'], epoch=epoch, max_epochs=config['training']['epochs'])
                ssl_model2.update_teacher(config['training']['momentum'], epoch=epoch, max_epochs=config['training']['epochs'])

            train_siamese_losses.append(siamese_loss.item())
            train_ssl_losses.append(ssl_loss.item())
            train_ssl_raw_losses.append((ssl_loss1.item() + ssl_loss2.item()) / 2)
            train_correct += torch.count_nonzero(y == (prob_sigmoid > 0.5)).item()
            train_total += len(y)

            current_siamese_loss = sum(train_siamese_losses) / len(train_siamese_losses)
            current_ssl_loss = sum(train_ssl_losses) / len(train_ssl_losses)
            current_ssl_raw_loss = sum(train_ssl_raw_losses) / len(train_ssl_raw_losses)
            current_acc = train_correct / train_total if train_total > 0 else 0
            train_tqdm.set_postfix(
                siamese_loss=f"{current_siamese_loss:.4f}",
                ssl_loss=f"{current_ssl_loss:.4f}",
                ssl_raw_loss=f"{current_ssl_raw_loss:.4f}",
                acc=f"{current_acc:.4f}",
                refresh=True
            )

        if scheduler is not None:
            scheduler.step()
        avg_train_siamese_loss = sum(train_siamese_losses) / len(train_siamese_losses) if train_siamese_losses else 0
        avg_train_ssl_loss = sum(train_ssl_losses) / len(train_ssl_losses) if train_ssl_losses else 0
        avg_train_ssl_raw_loss = sum(train_ssl_raw_losses) / len(train_ssl_raw_losses) if train_ssl_raw_losses else 0
        avg_train_acc = train_correct / train_total if train_total > 0 else 0
        logger.info(f"  Training Avg: Siamese Loss={avg_train_siamese_loss:.4f}, SSL Loss (Normalized)={avg_train_ssl_loss:.4f}, SSL Loss (Raw)={avg_train_ssl_raw_loss:.4f}, Accuracy={avg_train_acc:.4f}")

        siamese_model.eval()
        ssl_model1.eval()
        ssl_model2.eval()
        val_siamese_losses = []
        val_ssl_losses = []
        val_ssl_raw_losses = []
        val_correct = 0
        val_total = 0
        val_tqdm = tqdm(val_dataloader, desc=f'Epoch {epoch_num}/{config["training"]["epochs"]} Val', leave=False)

        with torch.no_grad():
            with autocast():
                for batch in val_tqdm:
                    try:
                        (img1, img2), y, _, _ = batch
                    except ValueError as e:
                        logger.error(f"Error unpacking batch during validation: {e}. Batch content: {batch}")
                        continue

                    img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])
                    img1_views = tensor_augment(img1, config)
                    img2_views = tensor_augment(img2, config)

                    prob_sigmoid, logits = siamese_model(img1, img2)
                    siamese_loss = siamese_criterion(logits, y)

                    outputs1 = ssl_model1(img1_views)
                    outputs2 = ssl_model2(img2_views)
                    s1_list = [s for s, _ in outputs1]
                    t1_list = [t for _, t in outputs1]
                    s2_list = [s for s, _ in outputs2]
                    t2_list = [t for _, t in outputs2]
                    
                    ssl_loss1 = dino_loss(s1_list, t1_list, config['loss']['student_temp'], config['loss']['teacher_temp'], config['loss']['center_momentum'])
                    ssl_loss2 = dino_loss(s2_list, t2_list, config['loss']['student_temp'], config['loss']['teacher_temp'], config['loss']['center_momentum'])

                    ssl_loss1_normalized = ssl_loss1 / dino_norm_factor
                    ssl_loss2_normalized = ssl_loss2 / dino_norm_factor
                    ssl_loss = (ssl_loss1_normalized + ssl_loss2_normalized) / 2

                    final_loss = config['loss']['lambda1'] * ssl_loss + config['loss']['lambda2'] * siamese_loss

                    val_siamese_losses.append(siamese_loss.item())
                    val_ssl_losses.append(ssl_loss.item())
                    val_ssl_raw_losses.append((ssl_loss1.item() + ssl_loss2.item()) / 2)
                    val_correct += torch.count_nonzero(y == (prob_sigmoid > 0.5)).item()
                    val_total += len(y)

                    current_siamese_loss = sum(val_siamese_losses) / len(val_siamese_losses)
                    current_ssl_loss = sum(val_ssl_losses) / len(val_ssl_losses)
                    current_ssl_raw_loss = sum(val_ssl_raw_losses) / len(val_ssl_raw_losses)
                    current_acc = val_correct / val_total if val_total > 0 else 0
                    val_tqdm.set_postfix(
                        siamese_loss=f"{current_siamese_loss:.4f}",
                        ssl_loss=f"{current_ssl_loss:.4f}",
                        ssl_raw_loss=f"{current_ssl_raw_loss:.4f}",
                        acc=f"{current_acc:.4f}",
                        refresh=True
                    )

        avg_val_siamese_loss = sum(val_siamese_losses) / len(val_siamese_losses) if val_siamese_losses else 0
        avg_val_ssl_loss = sum(val_ssl_losses) / len(val_ssl_losses) if val_ssl_losses else 0
        avg_val_ssl_raw_loss = sum(val_ssl_raw_losses) / len(val_ssl_raw_losses) if val_ssl_raw_losses else 0
        avg_val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = config['loss']['lambda1'] * avg_val_ssl_loss + config['loss']['lambda2'] * avg_val_siamese_loss
        logger.info(f"  Validation Avg: Siamese Loss={avg_val_siamese_loss:.4f}, SSL Loss (Normalized)={avg_val_ssl_loss:.4f}, SSL Loss (Raw)={avg_val_ssl_raw_loss:.4f}, Combined Loss={avg_val_loss:.4f}, Accuracy={avg_val_acc:.4f}")

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
                        "config": config
                    },
                    os.path.join(output_dir, "best.pth")
                )
            except Exception as e:
                logger.error(f"Error saving best model: {e}")

        if epoch_num % config['training']['save_after'] == 0:
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
                        "config": config
                    },
                    os.path.join(output_dir, f"epoch_{epoch_num}.pth")
                )
            except Exception as e:
                logger.error(f"Error saving checkpoint epoch_{epoch_num}: {e}")

    logger.info("Training finished.")

def main():
    parser = argparse.ArgumentParser(description="Train Siamese Network with DINO Integration")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['test_num'] = {'config1.yaml': 1, 'config2.yaml': 2, 'config3.yaml': 3}.get(os.path.basename(args.config), 1)
    
    run_training(config)

if __name__ == "__main__":
    main()