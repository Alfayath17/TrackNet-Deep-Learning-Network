import numpy as np
import pandas as pd
import os
import cv2
import argparse


# ============================================================
# Create Gaussian kernel
# ============================================================

def gaussian_kernel(size, variance):

    x, y = np.mgrid[
        -size:size + 1,
        -size:size + 1
    ]

    g = np.exp(
        -(x ** 2 + y ** 2) /
        float(2 * variance)
    )

    return g


# ============================================================
# Create normalized Gaussian image
# ============================================================

def create_gaussian(size, variance):

    gaussian_kernel_array = gaussian_kernel(
        size,
        variance
    )

    center = int(
        len(gaussian_kernel_array) / 2
    )

    gaussian_kernel_array = (
        gaussian_kernel_array *
        255 /
        gaussian_kernel_array[center][center]
    )

    gaussian_kernel_array = (
        gaussian_kernel_array.astype(np.uint8)
    )

    return gaussian_kernel_array


# ============================================================
# Create ground-truth heatmap images
# ============================================================

def create_gt_images(
    path_input,
    path_output,
    size,
    variance,
    width,
    height
):

    gaussian_kernel_array = create_gaussian(
        size,
        variance
    )

    # --------------------------------------------------------
    # Loop through games
    # --------------------------------------------------------

    for game_id in range(1, 11):

        game = 'game{}'.format(game_id)

        game_input_path = os.path.join(
            path_input,
            game
        )

        # Skip game if it does not exist
        if not os.path.isdir(game_input_path):

            print(
                'Skipping {} - folder not found'.format(
                    game_input_path
                )
            )

            continue

        clips = os.listdir(
            game_input_path
        )

        # ----------------------------------------------------
        # Loop through clips
        # ----------------------------------------------------

        for clip in clips:

            clip_input_path = os.path.join(
                game_input_path,
                clip
            )

            if not os.path.isdir(clip_input_path):
                continue

            print(
                'game = {}, clip = {}'.format(
                    game,
                    clip
                )
            )

            # ------------------------------------------------
            # Label file
            # ------------------------------------------------

            path_labels = os.path.join(
                clip_input_path,
                'Label.csv'
            )

            if not os.path.exists(path_labels):

                print(
                    'Label.csv not found: {}'.format(
                        path_labels
                    )
                )

                continue

            labels = pd.read_csv(
                path_labels
            )

            # ------------------------------------------------
            # Output:
            #
            # GT_Output/
            #     gts/
            #         game1/
            #             Clip1/
            # ------------------------------------------------

            path_out_clip = os.path.join(
                path_output,
                'gts',
                game,
                clip
            )

            os.makedirs(
                path_out_clip,
                exist_ok=True
            )

            # ------------------------------------------------
            # Create heatmap for every frame
            # ------------------------------------------------

            for idx in range(
                labels.shape[0]
            ):

                row = labels.iloc[idx]

                file_name = row['file name']
                vis = row['visibility']
                x = row['x-coordinate']
                y = row['y-coordinate']

                # --------------------------------------------
                # Empty heatmap
                # --------------------------------------------

                heatmap = np.zeros(
                    (
                        height,
                        width,
                        3
                    ),
                    dtype=np.uint8
                )

                # --------------------------------------------
                # If ball is visible
                # --------------------------------------------

                if pd.notna(vis) and vis != 0:

                    if pd.notna(x) and pd.notna(y):

                        x = int(x)
                        y = int(y)

                        # ------------------------------------
                        # Draw Gaussian
                        # ------------------------------------

                        for i in range(
                            -size,
                            size + 1
                        ):

                            for j in range(
                                -size,
                                size + 1
                            ):

                                px = x + i
                                py = y + j

                                # Check image boundaries
                                if (
                                    0 <= px < width
                                    and
                                    0 <= py < height
                                ):

                                    temp = gaussian_kernel_array[
                                        i + size,
                                        j + size
                                    ]

                                    if temp > 0:

                                        heatmap[
                                            py,
                                            px
                                        ] = (
                                            temp,
                                            temp,
                                            temp
                                        )

                # --------------------------------------------
                # Save GT heatmap
                # --------------------------------------------

                output_file = os.path.join(
                    path_out_clip,
                    str(file_name)
                )

                cv2.imwrite(
                    output_file,
                    heatmap
                )


# ============================================================
# Create training / validation CSV files
# ============================================================

def create_gt_labels(
    path_input,
    path_output,
    train_rate=0.7
):

    all_data = []

    # --------------------------------------------------------
    # Loop through games
    # --------------------------------------------------------

    for game_id in range(1, 11):

        game = 'game{}'.format(game_id)

        game_input_path = os.path.join(
            path_input,
            game
        )

        if not os.path.isdir(game_input_path):

            print(
                'Skipping {} - folder not found'.format(
                    game_input_path
                )
            )

            continue

        clips = os.listdir(
            game_input_path
        )

        # ----------------------------------------------------
        # Loop through clips
        # ----------------------------------------------------

        for clip in clips:

            clip_input_path = os.path.join(
                game_input_path,
                clip
            )

            if not os.path.isdir(
                clip_input_path
            ):
                continue

            path_labels = os.path.join(
                clip_input_path,
                'Label.csv'
            )

            if not os.path.exists(
                path_labels
            ):

                print(
                    'Label.csv not found: {}'.format(
                        path_labels
                    )
                )

                continue

            labels = pd.read_csv(
                path_labels
            )

            # ------------------------------------------------
            # Need at least 3 frames
            # ------------------------------------------------

            if len(labels) < 3:

                print(
                    'Skipping {} / {} - not enough frames'.format(
                        game,
                        clip
                    )
                )

                continue

            labels = labels.copy()

            # ------------------------------------------------
            # Image paths
            #
            # Actual structure:
            #
            # Dataset/
            #     game1/
            #         Clip1/
            #             0001.jpg
            #
            # Therefore:
            #
            # game1/Clip1/0001.jpg
            # ------------------------------------------------

            labels['path1'] = (
                game +
                '/' +
                clip +
                '/' +
                labels['file name'].astype(str)
            )

            # ------------------------------------------------
            # Ground-truth path
            #
            # GT_Output/
            #     gts/
            #         game1/
            #             Clip1/
            #                 0001.jpg
            # ------------------------------------------------

            labels['gt_path'] = (
                'gts/' +
                game +
                '/' +
                clip +
                '/' +
                labels['file name'].astype(str)
            )

            # ------------------------------------------------
            # Remove first two frames
            #
            # TrackNet needs:
            #
            # path1       = current frame
            # path2       = previous frame
            # path3       = previous-previous frame
            # ------------------------------------------------

            labels_target = labels.iloc[
                2:
            ].copy()

            # Current frame
            labels_target['path1'] = (
                labels['path1']
                .iloc[2:]
                .to_numpy()
            )

            # Previous frame
            labels_target['path2'] = (
                labels['path1']
                .iloc[1:-1]
                .to_numpy()
            )

            # Previous-previous frame
            labels_target['path3'] = (
                labels['path1']
                .iloc[:-2]
                .to_numpy()
            )

            # ------------------------------------------------
            # Select required columns
            # ------------------------------------------------

            labels_target = labels_target[
                [
                    'path1',
                    'path2',
                    'path3',
                    'gt_path',
                    'x-coordinate',
                    'y-coordinate',
                    'status',
                    'visibility'
                ]
            ]

            all_data.append(
                labels_target
            )

    # ========================================================
    # Combine all games and clips
    # ========================================================

    if len(all_data) == 0:

        raise RuntimeError(
            'No valid label data found. '
            'Check Dataset folder structure.'
        )

    df = pd.concat(
        all_data,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Shuffle dataset
    # --------------------------------------------------------

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    num_train = int(
        df.shape[0] *
        train_rate
    )

    df_train = df.iloc[
        :num_train
    ].copy()

    df_val = df.iloc[
        num_train:
    ].copy()

    # --------------------------------------------------------
    # Save CSV files
    # --------------------------------------------------------

    train_file = os.path.join(
        path_output,
        'labels_train.csv'
    )

    val_file = os.path.join(
        path_output,
        'labels_val.csv'
    )

    df_train.to_csv(
        train_file,
        index=False
    )

    df_val.to_csv(
        val_file,
        index=False
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print()
    print('==============================================')
    print('Ground-truth labels generated')
    print('==============================================')
    print(
        'Total samples      : {}'.format(
            len(df)
        )
    )
    print(
        'Training samples   : {}'.format(
            len(df_train)
        )
    )
    print(
        'Validation samples : {}'.format(
            len(df_val)
        )
    )
    print(
        'Train CSV          : {}'.format(
            train_file
        )
    )
    print(
        'Validation CSV     : {}'.format(
            val_file
        )
    )
    print('==============================================')


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

    # --------------------------------------------------------
    # TrackNet parameters
    # --------------------------------------------------------

    SIZE = 20
    VARIANCE = 10

    WIDTH = 1280
    HEIGHT = 720

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description='Generate TrackNet ground-truth data'
    )

    parser.add_argument(
        '--path_input',
        type=str,
        required=True,
        help='Path to Dataset folder'
    )

    parser.add_argument(
        '--path_output',
        type=str,
        required=True,
        help='Path to GT output folder'
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    os.makedirs(
        args.path_output,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create GT heatmap images
    # --------------------------------------------------------

    print()
    print('==============================================')
    print('Creating GT heatmap images')
    print('==============================================')

    create_gt_images(
        args.path_input,
        args.path_output,
        SIZE,
        VARIANCE,
        WIDTH,
        HEIGHT
    )

    # --------------------------------------------------------
    # Create training / validation labels
    # --------------------------------------------------------

    print()
    print('==============================================')
    print('Creating training / validation labels')
    print('==============================================')

    create_gt_labels(
        args.path_input,
        args.path_output
    )

    print()
    print('GT generation completed successfully.')