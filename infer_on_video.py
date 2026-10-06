from model import BallTrackerNet

import torch
import cv2
from general import postprocess
from tqdm import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance


def read_video(path_video):
    """
    Read video file.

    Returns:
        frames: list of video frames
        fps: frames per second
    """

    cap = cv2.VideoCapture(path_video)

    if not cap.isOpened():
        raise FileNotFoundError(
            "Cannot open video: {}".format(path_video)
        )

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps <= 0:
        fps = 30

    frames = []

    while cap.isOpened():

        ret, frame = cap.read()

        if ret:
            frames.append(frame)
        else:
            break

    cap.release()

    if len(frames) == 0:
        raise ValueError(
            "No frames could be read from video: {}".format(path_video)
        )

    print("Video frames :", len(frames))
    print("FPS          :", fps)
    print(
        "Resolution   : {} x {}".format(
            frames[0].shape[1],
            frames[0].shape[0]
        )
    )

    return frames, fps


def infer_model(frames, model, device):
    """
    Run TrackNet model on consecutive video frames.

    Returns:
        ball_track: detected ball coordinates
        dists: distances between consecutive detections
    """

    height = 360
    width = 640

    dists = [-1] * 2
    ball_track = [(None, None)] * 2

    for num in tqdm(
        range(2, len(frames)),
        desc="Detecting ball"
    ):

        img = cv2.resize(
            frames[num],
            (width, height)
        )

        img_prev = cv2.resize(
            frames[num - 1],
            (width, height)
        )

        img_preprev = cv2.resize(
            frames[num - 2],
            (width, height)
        )

        imgs = np.concatenate(
            (img, img_prev, img_preprev),
            axis=2
        )

        imgs = imgs.astype(
            np.float32
        ) / 255.0

        imgs = np.rollaxis(
            imgs,
            2,
            0
        )

        inp = np.expand_dims(
            imgs,
            axis=0
        )

        inp = torch.from_numpy(inp).float().to(device)

        with torch.no_grad():

            out = model(inp)

        output = (
            out.argmax(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        x_pred, y_pred = postprocess(output)

        ball_track.append(
            (x_pred, y_pred)
        )

        if (
            ball_track[-1][0] is not None
            and ball_track[-2][0] is not None
        ):

            dist = distance.euclidean(
                ball_track[-1],
                ball_track[-2]
            )

        else:

            dist = -1

        dists.append(dist)

    return ball_track, dists


def remove_outliers(
    ball_track,
    dists,
    max_dist=100
):
    """
    Remove sudden large jumps in ball prediction.
    """

    outliers = list(
        np.where(
            np.array(dists) > max_dist
        )[0]
    )

    for i in outliers:

        if i + 1 < len(dists):

            if (
                dists[i + 1] > max_dist
                or dists[i + 1] == -1
            ):

                ball_track[i] = (
                    None,
                    None
                )

        if i - 1 >= 0:

            if dists[i - 1] == -1:

                ball_track[i - 1] = (
                    None,
                    None
                )

    return ball_track


def split_track(
    ball_track,
    max_gap=4,
    max_dist_gap=80,
    min_track=5
):
    """
    Split ball track into subtracks for interpolation.
    """

    list_det = [
        0 if x[0] is not None else 1
        for x in ball_track
    ]

    groups = [
        (k, sum(1 for _ in g))
        for k, g in groupby(list_det)
    ]

    cursor = 0
    min_value = 0
    result = []

    for i, (k, length) in enumerate(groups):

        if (
            k == 1
            and i > 0
            and i < len(groups) - 1
        ):

            if (
                ball_track[cursor - 1][0] is not None
                and ball_track[cursor + length][0] is not None
            ):

                dist = distance.euclidean(
                    ball_track[cursor - 1],
                    ball_track[cursor + length]
                )

                if (
                    length >= max_gap
                    or dist / length > max_dist_gap
                ):

                    if cursor - min_value > min_track:

                        result.append(
                            [min_value, cursor]
                        )

                        min_value = (
                            cursor + length - 1
                        )

        cursor += length

    if len(list_det) - min_value > min_track:

        result.append(
            [min_value, len(list_det)]
        )

    return result


def interpolation(coords):
    """
    Interpolate missing ball coordinates.
    """

    def nan_helper(y):
        return (
            np.isnan(y),
            lambda z: z.nonzero()[0]
        )

    x = np.array([
        point[0]
        if point[0] is not None
        else np.nan
        for point in coords
    ])

    y = np.array([
        point[1]
        if point[1] is not None
        else np.nan
        for point in coords
    ])

    nans, index = nan_helper(x)

    if np.any(~nans):

        x[nans] = np.interp(
            index(nans),
            index(~nans),
            x[~nans]
        )

    nans, index = nan_helper(y)

    if np.any(~nans):

        y[nans] = np.interp(
            index(nans),
            index(~nans),
            y[~nans]
        )

    track = list(
        zip(x, y)
    )

    return track


def write_track(
    frames,
    ball_track,
    path_output_video,
    fps,
    trace=7
):
    """
    Write output video with ball tracking overlay.
    """

    height, width = frames[0].shape[:2]

    out = cv2.VideoWriter(
        path_output_video,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    if not out.isOpened():

        raise RuntimeError(
            "Could not create output video: {}".format(
                path_output_video
            )
        )

    for num in tqdm(
        range(len(frames)),
        desc="Writing output video"
    ):

        frame = frames[num].copy()

        for i in range(trace):

            if num - i >= 0:

                point = ball_track[num - i]

                if point[0] is not None:

                    x = int(point[0])
                    y = int(point[1])

                    frame = cv2.circle(
                        frame,
                        (x, y),
                        radius=0,
                        color=(0, 0, 255),
                        thickness=max(
                            1,
                            10 - i
                        )
                    )

                else:

                    break

        out.write(frame)

    out.release()

    print(
        "Output video saved to: {}".format(
            path_output_video
        )
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='path to trained model'
    )

    parser.add_argument(
        '--video_path',
        type=str,
        required=True,
        help='path to input video'
    )

    parser.add_argument(
        '--video_out_path',
        type=str,
        required=True,
        help='path to output video'
    )

    parser.add_argument(
        '--extrapolation',
        action='store_true',
        help='whether to interpolate missing ball detections'
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    print("=" * 60)
    print("TrackNet Video Inference")
    print("=" * 60)

    print(
        "PyTorch version :",
        torch.__version__
    )

    print(
        "Using device    :",
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
    # LOAD MODEL
    # --------------------------------------------------

    model = BallTrackerNet()

    print(
        "Loading model:",
        args.model_path
    )

    model.load_state_dict(
        torch.load(
            args.model_path,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------
    # READ VIDEO
    # --------------------------------------------------

    frames, fps = read_video(
        args.video_path
    )

    # --------------------------------------------------
    # INFERENCE
    # --------------------------------------------------

    ball_track, dists = infer_model(
        frames,
        model,
        device
    )

    # --------------------------------------------------
    # REMOVE OUTLIERS
    # --------------------------------------------------

    ball_track = remove_outliers(
        ball_track,
        dists
    )

    # --------------------------------------------------
    # OPTIONAL INTERPOLATION
    # --------------------------------------------------

    if args.extrapolation:

        print(
            "Running ball-track interpolation..."
        )

        subtracks = split_track(
            ball_track
        )

        for r in subtracks:

            ball_subtrack = ball_track[
                r[0]:r[1]
            ]

            ball_subtrack = interpolation(
                ball_subtrack
            )

            ball_track[
                r[0]:r[1]
            ] = ball_subtrack

    # --------------------------------------------------
    # WRITE OUTPUT
    # --------------------------------------------------

    write_track(
        frames,
        ball_track,
        args.video_out_path,
        fps
    )

    print("=" * 60)
    print("Inference completed successfully.")
    print("=" * 60)