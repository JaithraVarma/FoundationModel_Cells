import logging
import os
import argparse
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter # Keep commented for now
from torch import nn
# Enable AMP (Automatic Mixed Precision)
from torch.cuda.amp import GradScaler, autocast 
# Enable CuDNN benchmarking
import torch.backends.cudnn as cudnn

# Updated imports for the new structure
# Assuming src directory is in PYTHONPATH or script is run from project root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/nfs/tier2/users/sm1367/Cell_Model')))
from src.models.siamese_network import SiameseNetwork 
from src.data.dataset_new import PairDataset 

logger = logging.getLogger(__name__)
# Basic logger configuration (can be enhanced later)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')


def run_training(args):
    """Runs the Siamese Network training process."""
    logger.info(f"Starting training process with args: {args}")

    # --- Optimization Flags ---
    cudnn.benchmark = True
    logger.info(f"torch.backends.cudnn.benchmark set to {cudnn.benchmark}")

    # --- Device Setup ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # --- Output Directory and Logging Setup ---
    # Construct output path based on parameters - Use os.path.join correctly
    run_name = f"{args.backbone}_Adam_lr_{args.learning_rate}_bs_{args.batch_size}_e_{args.epochs}"
    output_dir = os.path.join(args.out_path, run_name)
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, 'train.log')
    
    # Setup file logging within the output directory
    # Remove existing handlers if any before adding new one
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
             logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler) # Add file handler
    logger.info(f"Created output directory: {output_dir}")
    logger.info(f"Logging to {log_file_path}")

    # Save args to files
    try:
        with open(os.path.join(output_dir, "args.txt"), "w") as f:
            f.write(str(vars(args))) # Use vars(args) for cleaner output
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            yaml.dump(vars(args), f)
        logger.info("Saved arguments to args.txt and args.yaml")
    except Exception as e:
        logger.error(f"Error saving arguments: {e}")


    # --- Dataset and DataLoader ---
    logger.info("Setting up datasets and dataloaders...")
    try:
        # Assuming PairDataset needs path, shuffle_pairs, augment, num_workers
        train_dataset = PairDataset(args.train_path, shuffle_pairs=True, augment=True, num_workers=args.num_workers)
        val_dataset = PairDataset(args.val_path, shuffle_pairs=False, augment=False, num_workers=args.num_workers)
        
        # Dataloaders - shuffling handled here for training set
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_workers, shuffle=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
        logger.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
        logger.info(f"Train dataloader batches: {len(train_dataloader)}, Val dataloader batches: {len(val_dataloader)}")
    except Exception as e:
        logger.exception(f"Error creating datasets/dataloaders: {e}") # Use logger.exception for traceback
        return # Stop execution if data fails

    # --- Model ---
    logger.info(f"Initializing SiameseNetwork with backbone: {args.backbone}")
    try:
        model = SiameseNetwork(backbone=args.backbone)
        # logger.info(model) # Log model structure if needed (can be verbose)
        
        # Move model to device(s)
        model.to(device) # Move to primary device first

        # Check for multiple GPUs and wrap with DataParallel
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} CUDA devices.")
        if gpu_count > 1:
            logger.info(f"Using nn.DataParallel for {gpu_count} GPUs.")
            model = nn.DataParallel(model) # Wrap the model

    except Exception as e:
        logger.exception(f"Error initializing model: {e}")
        return

    # --- Optimizer and Criterion ---
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate) # Original SGD option
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    logger.info(f"Optimizer: Adam, LR: {args.learning_rate}")
    logger.info(f"Criterion: BCEWithLogitsLoss")

    # --- AMP Scaler ---
    scaler = GradScaler()
    logger.info("Initialized GradScaler for AMP.")

    # --- Training Loop ---
    best_val_loss = float('inf')
    logger.info(f"Starting training for {args.epochs} epochs.")

    for epoch in range(args.epochs):
        epoch_num = epoch + 1
        logger.info(f"--- Epoch [{epoch_num}/{args.epochs}] ---")

        # Pair creation should happen automatically inside PairDataset if designed that way,
        # or handled by DataLoader shuffling if pairs are static per image.
        # The original train.py called dataset.create_pairs() each epoch - confirm if this is necessary.
        # If pairs need dynamic regeneration each epoch AND it's not handled by the Dataset internally,
        # this might require adjustments (e.g., custom sampler or modifying dataset).
        # For now, assuming shuffle=True in DataLoader is sufficient if pairs are tied to index.

        ## --- Training Phase ---
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        # Use tqdm for progress bar
        train_tqdm_instance = tqdm(train_dataloader, desc=f'Epoch {epoch_num}/{args.epochs} Train', leave=False)

        for batch in train_tqdm_instance:
            # Ensure batch unpacking matches dataset's __getitem__ return format
            try:
                # Original dataset returns: (img1, img2), target, (class1, class2), (path1, path2)
                (img1, img2), y, _, _ = batch # Unpack only needed items
            except ValueError as e:
                 logger.error(f"Error unpacking batch: {e}. Batch content: {batch}")
                 continue # Skip batch if format is wrong
                 
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            # Forward pass with autocast for AMP
            with autocast(): 
                prob_sigmoid, logits = model(img1, img2) # Assuming model returns (sigmoid_prob, raw_logits)
                loss = criterion(logits, y) # Use raw logits with BCEWithLogitsLoss

            # Backward pass and optimization with scaler
            optimizer.zero_grad()
            scaler.scale(loss).backward() # Scale loss
            scaler.step(optimizer) # Unscale gradients and step optimizer
            scaler.update() # Update scaler for next iteration

            # Logging and Metrics
            train_losses.append(loss.item())
            # Calculate accuracy using sigmoid probability > 0.5 threshold
            train_correct += torch.count_nonzero(y == (prob_sigmoid > 0.5)).item()
            train_total += len(y)
            
            # Update tqdm description
            current_loss = sum(train_losses) / len(train_losses)
            current_acc = train_correct / train_total if train_total > 0 else 0
            train_tqdm_instance.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}", refresh=True)

        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
        avg_train_acc = train_correct / train_total if train_total > 0 else 0
        # writer.add_scalar('train_loss', avg_train_loss, epoch)
        # writer.add_scalar('train_acc', avg_train_acc, epoch)
        logger.info(f"  Training Avg: Loss={avg_train_loss:.4f}, Accuracy={avg_train_acc:.4f}")


        ## --- Validation Phase ---
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        val_tqdm_instance = tqdm(val_dataloader, desc=f'Epoch {epoch_num}/{args.epochs} Val', leave=False)

        with torch.no_grad(): # Disable gradient calculations for validation
            # Still use autocast for validation for consistency and potential speedup
            with autocast():
                for batch in val_tqdm_instance:
                    try:
                        (img1, img2), y, _, _ = batch # Unpack only needed items
                    except ValueError as e:
                         logger.error(f"Error unpacking batch during validation: {e}. Batch content: {batch}")
                         continue
                         
                    img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

                    prob_sigmoid, logits = model(img1, img2)
                    loss = criterion(logits, y)

                    val_losses.append(loss.item())
                    val_correct += torch.count_nonzero(y == (prob_sigmoid > 0.5)).item()
                    val_total += len(y)
                    
                    current_val_loss = sum(val_losses) / len(val_losses)
                    current_val_acc = val_correct / val_total if val_total > 0 else 0
                    val_tqdm_instance.set_postfix(loss=f"{current_val_loss:.4f}", acc=f"{current_val_acc:.4f}", refresh=True)

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        avg_val_acc = val_correct / val_total if val_total > 0 else 0
        # writer.add_scalar('val_loss', avg_val_loss, epoch)
        # writer.add_scalar('val_acc', avg_val_acc, epoch)
        logger.info(f"  Validation Avg: Loss={avg_val_loss:.4f}, Accuracy={avg_val_acc:.4f}")


        ## --- Model Checkpointing ---
        # Update "best.pth" model if val_loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"  New best validation loss: {best_val_loss:.4f}. Saving best model...")
            try:
                torch.save(
                    {
                        "epoch": epoch_num,
                        "model_state_dict": model.state_dict(),
                        "backbone": args.backbone,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "args": vars(args) # Save args used for this model
                    },
                    os.path.join(output_dir, "best.pth")
                )
            except Exception as e:
                logger.error(f"Error saving best model: {e}")

        # Save model periodically based on "args.save_after"
        if epoch_num % args.save_after == 0:
             logger.info(f"  Saving checkpoint model at epoch {epoch_num}...")
             try:
                 torch.save(
                     {
                         "epoch": epoch_num,
                         "model_state_dict": model.state_dict(),
                         "backbone": args.backbone,
                         "optimizer_state_dict": optimizer.state_dict(),
                         "val_loss": avg_val_loss, # Store current val loss
                         "args": vars(args)
                     },
                     os.path.join(output_dir, f"epoch_{epoch_num}.pth")
                 )
             except Exception as e:
                 logger.error(f"Error saving checkpoint model epoch_{epoch_num}: {e}")

    # --- End of Training ---
    logger.info("Training finished.")
    # if writer: writer.close()


if __name__ == "__main__":
    
    # --- Set CUDA device ---
    # Add this line to specify GPU 1
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1' # REMOVED to allow multi-GPU
    # print(f"INFO: Setting CUDA_VISIBLE_DEVICES to '1'") # Add print for confirmation

    # Separate argument parsing from main logic
    parser = argparse.ArgumentParser(description="Train Siamese Network based on src/models/train_model.py")

    # Arguments copied and adapted from original train.py
    parser.add_argument(
        '--train_path',
        type=str,
        help="Path to CSV file containing training dataset info (path, class, imagescore).",
        required=False, # Reverted to False as per user request
        default='/nfs/tier2/users/sm1367/Cell_Model/csv_creator/2d_ctc_train_dataset.csv' # Added default path
    )
    parser.add_argument(
        '--val_path',
        type=str,
        help="Path to CSV file containing validation dataset info.",
        required=False, # Reverted to False as per user request
        default='/nfs/tier2/users/sm1367/Cell_Model/csv_creator/2d_ctc_test_dataset.csv' # Added default path
    )
    parser.add_argument(
        '-o', '--out_path',
        type=str,
        help="Base path for outputting model weights, logs, etc.",
        default='outputs/training_runs/' # Sensible default relative path
    )
    parser.add_argument(
        '-b', '--backbone',
        type=str,
        help="Network backbone (e.g., 'vit_b_16', 'resnet50').",
        default="vit_b_16"
    )
    parser.add_argument(
        '-lr', '--learning_rate',
        type=float,
        help="Learning Rate",
        default=5e-6
    )
    parser.add_argument(
        '-bs', '--batch_size',
        type=int,
        help="Batch Size",
        default=32
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        help="Number of epochs to train",
        default=100
    )
    parser.add_argument(
        '-s', '--save_after',
        type=int,
        help="Save model checkpoint after every N epochs.",
        default=1
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        help="Number of workers for DataLoader.",
        default=8 # Increased default from 4
    )
    # Example config file argument (optional)
    # parser.add_argument('--config', type=str, help='Path to YAML config file to override CLI args', default=None)

    args = parser.parse_args()
    
    # --- Start Logging for __main__ execution ---
    # Setup basic console logging for this direct execution context
    # The run_training function will add its file handler
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)
    # Clear existing handlers if re-running in same session (e.g., notebook)
    for handler in main_logger.handlers[:]:
        main_logger.removeHandler(handler)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
    console_handler.setFormatter(console_formatter)
    main_logger.addHandler(console_handler)
    
    logger.info("Running train_model.py script directly.")
    logger.info(f"Attempting to use GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}") # Log the setting
    logger.info(f"Parsed arguments: {args}")

    # Check if required paths exist
    if not os.path.exists(args.train_path):
        logger.error(f"Training data path not found: {args.train_path}")
        exit(1)
    if not os.path.exists(args.val_path):
        logger.error(f"Validation data path not found: {args.val_path}")
        exit(1)

    # Call the main training function
    try:
        run_training(args)
    except Exception as e:
        logger.exception(f"An uncaught error occurred during training: {e}")
        exit(1)
