import torch
import time
import numpy as np
import torch.nn as nn
import cv2
from scipy.spatial import distance


def train(model, train_loader, optimizer, device, epoch, max_iters=200):
    """
    Train the TrackNet model for one epoch.
    """

    start_time = time.time()
    losses = []

    criterion = nn.CrossEntropyLoss()

    model.train()

    for iter_id, batch in enumerate(train_loader):

        # Stop after max_iters iterations
        if iter_id >= max_iters:
            break

        optimizer.zero_grad()

        # Input images
        inputs = batch[0].float().to(device)

        # Ground-truth heatmap
        gt = batch[1].long().to(device)

        # Forward pass
        out = model(inputs)

        # Calculate loss
        loss = criterion(out, gt)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Time information
        elapsed_time = time.time() - start_time
        duration = time.strftime(
            "%H:%M:%S",
            time.gmtime(elapsed_time)
        )

        print(
            "train | epoch = {}, iter = [{}|{}], loss = {}, time = {}".format(
                epoch,
                iter_id + 1,
                max_iters,
                round(loss.item(), 6),
                duration
            )
        )

        losses.append(loss.item())

    if len(losses) == 0:
        return 0.0

    return np.mean(losses)


def validate(model, val_loader, device, epoch, min_dist=5):

    """
    Validate the TrackNet model.

    min_dist:
        Maximum pixel distance between predicted ball position
        and ground-truth position for a correct detection.
    """

    losses = []

    # TP / FP / TN / FN
    # Index:
    # 0 = invisible
    # 1,2,3 = different visibility/status classes
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]

    criterion = nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():

        for iter_id, batch in enumerate(val_loader):

            # Input
            inputs = batch[0].float().to(device)

            # Ground truth
            gt = batch[1].long().to(device)

            # Forward pass
            out = model(inputs)

            # Loss
            loss = criterion(out, gt)
            losses.append(loss.item())

            # Get predicted class for every pixel
            output = out.argmax(dim=1).cpu().numpy()

            for i in range(len(output)):

                # Convert heatmap to ball coordinates
                x_pred, y_pred = postprocess(output[i])

                # Convert DataLoader tensors to Python numbers
                x_gt = float(batch[2][i].item())
                y_gt = float(batch[3][i].item())
                vis = int(batch[4][i].item())

                # Safety check for visibility index
                if vis < 0 or vis >= len(tp):
                    print(
                        "Warning: invalid visibility value: {}".format(vis)
                    )
                    continue

                # -------------------------------------------------
                # Ball detected by model
                # -------------------------------------------------

                if x_pred is not None:

                    # Ground truth says ball is visible
                    if vis != 0:

                        dst = distance.euclidean(
                            (x_pred, y_pred),
                            (x_gt, y_gt)
                        )

                        # Correct prediction
                        if dst < min_dist:
                            tp[vis] += 1

                        # Prediction exists but position is wrong
                        else:
                            fp[vis] += 1

                    # Model detected ball but ground truth is invisible
                    else:
                        fp[vis] += 1

                # -------------------------------------------------
                # No ball detected by model
                # -------------------------------------------------

                else:

                    # Ball should be visible
                    if vis != 0:
                        fn[vis] += 1

                    # Correctly detected no ball
                    else:
                        tn[vis] += 1

    # -------------------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------------------

    eps = 1e-15

    total_tp = sum(tp)
    total_fp = sum(fp)
    total_tn = sum(tn)
    total_fn = sum(fn)

    precision = total_tp / (
        total_tp + total_fp + eps
    )

    recall = total_tp / (
        total_tp + total_fn + eps
    )

    f1 = (
        2 * precision * recall /
        (precision + recall + eps)
    )

    mean_loss = np.mean(losses) if losses else 0.0

    print(
        "validate | epoch = {}, loss = {:.6f}, "
        "precision = {:.6f}, recall = {:.6f}, F1 = {:.6f}".format(
            epoch,
            mean_loss,
            precision,
            recall,
            f1
        )
    )

    print(
        "TP = {}, FP = {}, TN = {}, FN = {}".format(
            total_tp,
            total_fp,
            total_tn,
            total_fn
        )
    )

    return mean_loss, precision, recall, f1


def postprocess(feature_map, scale=2):

    """
    Convert TrackNet output heatmap into ball coordinates.

    TrackNet input:
        640 x 360

    Original video:
        1280 x 720

    Therefore:
        scale = 2
    """

    # Make a copy so the original array is not modified
    feature_map = feature_map.copy()

    # Convert to 0-255
    feature_map = feature_map * 255

    # Reshape to TrackNet resolution
    feature_map = feature_map.reshape((360, 640))

    # Convert to uint8
    feature_map = feature_map.astype(np.uint8)

    # Threshold
    ret, heatmap = cv2.threshold(
        feature_map,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Detect circle representing the ball
    circles = cv2.HoughCircles(
        heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=2,
        minRadius=2,
        maxRadius=7
    )

    x = None
    y = None

    if circles is not None:

        if len(circles) == 1:

            x = circles[0][0][0] * scale
            y = circles[0][0][1] * scale

    return x, y