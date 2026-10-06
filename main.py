from model import BallTrackerNet
import torch
from datasets import trackNetDataset
import torch.optim as optim
import os
from tensorboardX import SummaryWriter
from general import train, validate
import argparse


if __name__ == '__main__':

    # ---------------------------------------------------------
    # Arguments
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='batch size'
    )

    parser.add_argument(
        '--exp_id',
        type=str,
        default='default',
        help='experiment name'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=500,
        help='total training epochs'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1.0,
        help='learning rate'
    )

    parser.add_argument(
        '--val_intervals',
        type=int,
        default=5,
        help='number of epochs between validation'
    )

    parser.add_argument(
        '--steps_per_epoch',
        type=int,
        default=200,
        help='number of training steps per epoch'
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Select CPU or GPU automatically
    # ---------------------------------------------------------
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("=" * 60)
    print("TrackNet Training")
    print("=" * 60)
    print("PyTorch version :", torch.__version__)
    print("Using device    :", device)

    if torch.cuda.is_available():
        print("GPU             :", torch.cuda.get_device_name(0))
        print("CUDA version    :", torch.version.cuda)
    else:
        print("GPU             : Not available")
        print("Training will run on CPU")

    print("=" * 60)

    # ---------------------------------------------------------
    # Training dataset
    # ---------------------------------------------------------
    train_dataset = trackNetDataset('train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # ---------------------------------------------------------
    # Validation dataset
    # ---------------------------------------------------------
    val_dataset = trackNetDataset('val')

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print("Training samples  :", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Batch size        :", args.batch_size)
    print("Steps per epoch   :", args.steps_per_epoch)
    print("Epochs            :", args.num_epochs)
    print("=" * 60)

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------
    model = BallTrackerNet()

    model = model.to(device)

    print("Model loaded successfully.")
    print("=" * 60)

    # ---------------------------------------------------------
    # Experiment directories
    # ---------------------------------------------------------
    exps_path = os.path.join(
        './exps',
        args.exp_id
    )

    tb_path = os.path.join(
        exps_path,
        'plots'
    )

    os.makedirs(tb_path, exist_ok=True)

    # ---------------------------------------------------------
    # TensorBoard
    # ---------------------------------------------------------
    log_writer = SummaryWriter(tb_path)

    # ---------------------------------------------------------
    # Model save paths
    # ---------------------------------------------------------
    model_last_path = os.path.join(
        exps_path,
        'model_last.pt'
    )

    model_best_path = os.path.join(
        exps_path,
        'model_best.pt'
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    optimizer = optim.Adadelta(
        model.parameters(),
        lr=args.lr
    )

    # Best validation F1
    val_best_metric = 0.0

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    for epoch in range(args.num_epochs):

        print("\n")
        print("=" * 60)
        print("Epoch {}/{}".format(
            epoch + 1,
            args.num_epochs
        ))
        print("=" * 60)

        # -----------------------------------------------------
        # Training
        # -----------------------------------------------------
        train_loss = train(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.steps_per_epoch
        )

        print(
            "train loss = {}".format(
                train_loss
            )
        )

        # TensorBoard
        log_writer.add_scalar(
            'Train/training_loss',
            train_loss,
            epoch
        )

        log_writer.add_scalar(
            'Train/lr',
            optimizer.param_groups[0]['lr'],
            epoch
        )

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------
        if epoch > 0 and epoch % args.val_intervals == 0:

            val_loss, precision, recall, f1 = validate(
                model,
                val_loader,
                device,
                epoch
            )

            print(
                "val loss  = {}".format(
                    val_loss
                )
            )

            print(
                "precision  = {}".format(
                    precision
                )
            )

            print(
                "recall     = {}".format(
                    recall
                )
            )

            print(
                "F1         = {}".format(
                    f1
                )
            )

            # TensorBoard
            log_writer.add_scalar(
                'Val/loss',
                val_loss,
                epoch
            )

            log_writer.add_scalar(
                'Val/precision',
                precision,
                epoch
            )

            log_writer.add_scalar(
                'Val/recall',
                recall,
                epoch
            )

            log_writer.add_scalar(
                'Val/f1',
                f1,
                epoch
            )

            # -------------------------------------------------
            # Save best model
            # -------------------------------------------------
            if f1 > val_best_metric:

                val_best_metric = f1

                torch.save(
                    model.state_dict(),
                    model_best_path
                )

                print(
                    "Best model saved: {}".format(
                        model_best_path
                    )
                )

        # -----------------------------------------------------
        # Save latest model every epoch
        # -----------------------------------------------------
        torch.save(
            model.state_dict(),
            model_last_path
        )

        print(
            "Latest model saved: {}".format(
                model_last_path
            )
        )

    # ---------------------------------------------------------
    # Finish
    # ---------------------------------------------------------
    log_writer.close()

    print("\n")
    print("=" * 60)
    print("Training completed.")
    print("Best model :", model_best_path)
    print("Last model :", model_last_path)
    print("=" * 60)