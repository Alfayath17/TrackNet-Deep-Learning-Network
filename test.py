from model import BallTrackerNet
import torch
from datasets import trackNetDataset
from general import validate
import argparse
import os


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a trained TrackNet model"
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='validation batch size'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='path to trained TrackNet model'
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("=" * 60)
    print("TrackNet Model Evaluation")
    print("=" * 60)

    print(
        "PyTorch version :",
        torch.__version__
    )

    print(
        "Device          :",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU             :",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "GPU             : Not available"
        )

    print("=" * 60)

    # --------------------------------------------------
    # CHECK MODEL FILE
    # --------------------------------------------------

    if not os.path.exists(args.model_path):

        raise FileNotFoundError(
            "Model file not found: {}".format(
                args.model_path
            )
        )

    print(
        "Model path      :",
        args.model_path
    )

    # --------------------------------------------------
    # LOAD VALIDATION DATASET
    # --------------------------------------------------

    val_dataset = trackNetDataset('val')

    print(
        "Validation samples:",
        len(val_dataset)
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    model = BallTrackerNet()

    checkpoint = torch.load(
        args.model_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model = model.to(device)

    model.eval()

    print("Model loaded successfully.")
    print("=" * 60)

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    print("Running validation...")
    print("This may take some time on CPU.")
    print("=" * 60)

    val_loss, precision, recall, f1 = validate(
        model,
        val_loader,
        device,
        -1
    )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("TrackNet Validation Results")
    print("=" * 60)

    print(
        "Validation Loss : {:.6f}".format(
            val_loss
        )
    )

    print(
        "Precision       : {:.6f}".format(
            precision
        )
    )

    print(
        "Recall          : {:.6f}".format(
            recall
        )
    )

    print(
        "F1 Score        : {:.6f}".format(
            f1
        )
    )

    print("=" * 60)


if __name__ == '__main__':
    main()