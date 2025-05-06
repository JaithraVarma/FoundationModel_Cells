import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import argparse

import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from siamese import SiameseNetwork
from libs.dataset_new import Dataset, PairDataset



if __name__ == "__main__":
    logger.info("Training Siamese Network")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_path',
        type=str,
        help="Path to directory containing training dataset.",
        required=False,
        default='/nfs/tier1/users/shk35/projects/embryo_witnessing/data/txt_files/train.csv'
    )
    parser.add_argument(
        '--val_path',
        type=str,
        help="Path to directory containing validation dataset.",
        required=False,
        default='/nfs/tier1/users/shk35/projects/embryo_witnessing/data/txt_files/validation.csv'
    )
    parser.add_argument(
        '-o',
        '--out_path',
        type=str,
        help="Path for outputting model weights and tensorboard summary.",
        required=False,
        default='../models/siamese/vit_b_16/CF/v1/'
    )
    parser.add_argument(
        '-b',
        '--backbone',
        type=str,
        help="Network backbone from torchvision.models to be used in the siamese network.",
        default="vit_b_16"
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        help="Learning Rate",
        default=5e-6
    )
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        help="Batch Size",
        default=32
    )
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        help="Number of epochs to train",
        default=100
    )
    parser.add_argument(
        '-s',
        '--save_after',
        type=int,
        help="Model checkpoint is saved after each specified number of epochs.",
        default=1
    )

    args = parser.parse_args()
    args.out_path = os.path.join(args.out_path, args.backbone + "_Adam_lr_" + str(args.learning_rate)
                                 + "_bs_" + str(args.batch_size) + "_e_" + str(args.epochs))
    os.makedirs(args.out_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.out_path, 'train.log'), level=logging.INFO)


    # save args to a file
    with open(os.path.join(args.out_path, "args.txt"), "w") as f:
        f.write(str(args))
    # save to yaml file
    with open(os.path.join(args.out_path, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    logger.info(f"Training Siamese Network with args: {args}")
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_dataset = PairDataset(args.train_path, shuffle_pairs=True, augment=True, num_workers=16)
    val_dataset = PairDataset(args.val_path, shuffle_pairs=False, augment=False, num_workers=16)
    logger.info('DataLaoder for Training Set')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=16)
    logger.info('DataLaoder for Validation Set')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=16)

    model = SiameseNetwork(backbone=args.backbone)
    # logger.info model summary
    logger.info(model)
    model.to(device)

    # # model.load_state_dict(model_state_dict)

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    # writer = SummaryWriter(os.path.join(args.out_path, "summary"))

    best_val = 10000000000

    for epoch in range(args.epochs):

        logger.info("[{} / {}]".format(epoch, args.epochs))
        train_dataset.create_pairs()
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      drop_last=True, num_workers=16, shuffle=True)

        val_dataset.create_pairs()
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=16)

        model.train()

        losses = []
        correct = 0
        total = 0

        logger.info(f"len of train_dataloader: {len(train_dataloader)}")
        logger.info(f"len of val_dataloader: {len(val_dataloader)}")
        tqdm_instance = tqdm(train_dataloader, desc=f'Epoch{epoch}/{args.epochs}', total=len(train_dataloader))

        # Training Loop Start
        for (img1, img2), y, (class1, class2), _ in tqdm_instance:
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)[1]
            loss = criterion(prob, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)
            tqdm_instance.set_description(desc=f"Training: Loss={sum(losses) / len(losses):.2f}"
                                               f"\t Train Acc={correct / total:.2f}", refresh=True)

        # writer.add_scalar('train_loss', sum(losses)/len(losses), epoch)
        # writer.add_scalar('train_acc', correct / total, epoch)

        logger.info("\tTraining: Loss={:.2f}\t Avg Train Accuracy={:.2f}\t".format(sum(losses) / len(losses), correct / total))
        # Training Loop End

        # Evaluation Loop Start
        model.eval()

        losses = []
        correct = 0
        total = 0

        for (img1, img2), y, (class1, class2), _ in val_dataloader:
            img1, img2, y = map(lambda x: x.to(device), [img1, img2, y])

            prob = model(img1, img2)[1]
            loss = criterion(prob, y)

            losses.append(loss.item())
            correct += torch.count_nonzero(y == (prob > 0.5)).item()
            total += len(y)

        val_loss = sum(losses) / max(1, len(losses))
        # writer.add_scalar('val_loss', val_loss, epoch)
        # writer.add_scalar('val_acc', correct / total, epoch)

        logger.info("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, correct / total))
        # Evaluation Loop End

        # Update "best.pth" model if val_loss in current epoch is lower than the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "best.pth")
            )

        # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % args.save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, "epoch_{}.pth".format(epoch + 1))
            )
