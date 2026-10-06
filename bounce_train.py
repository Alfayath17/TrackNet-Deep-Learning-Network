import os
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import catboost as ctb


# ============================================================
# CREATE BOUNCE FEATURES
# ============================================================

def create_features(path_dataset, num_frames=3):

    print("=" * 60)
    print("Creating bounce detection features")
    print("=" * 60)

    if not os.path.exists(path_dataset):

        raise FileNotFoundError(
            "Dataset folder not found: {}".format(
                path_dataset
            )
        )

    all_data = []

    games = sorted(
        [
            x for x in os.listdir(path_dataset)
            if os.path.isdir(
                os.path.join(path_dataset, x)
            )
        ]
    )

    print(
        "Games found:",
        len(games)
    )

    for game in tqdm(
        games,
        desc="Processing games"
    ):

        game_path = os.path.join(
            path_dataset,
            game
        )

        clips = sorted(
            [
                x for x in os.listdir(game_path)
                if os.path.isdir(
                    os.path.join(game_path, x)
                )
            ]
        )

        for clip in clips:

            clip_path = os.path.join(
                game_path,
                clip
            )

            label_path = os.path.join(
                clip_path,
                'Label.csv'
            )

            if not os.path.exists(label_path):

                print(
                    "\nSkipping missing Label.csv:",
                    label_path
                )

                continue

            labels = pd.read_csv(
                label_path
            )

            required_columns = [
                'x-coordinate',
                'y-coordinate',
                'status'
            ]

            missing_columns = [
                col
                for col in required_columns
                if col not in labels.columns
            ]

            if missing_columns:

                print(
                    "\nSkipping {} because columns are missing: {}".format(
                        label_path,
                        missing_columns
                    )
                )

                continue

            labels = labels.copy()

            # ------------------------------------------------
            # Convert coordinate/status columns to numeric
            # ------------------------------------------------

            labels['x-coordinate'] = pd.to_numeric(
                labels['x-coordinate'],
                errors='coerce'
            )

            labels['y-coordinate'] = pd.to_numeric(
                labels['y-coordinate'],
                errors='coerce'
            )

            labels['status'] = pd.to_numeric(
                labels['status'],
                errors='coerce'
            )

            # ------------------------------------------------
            # Create temporal movement features
            # ------------------------------------------------

            eps = 1e-15

            for i in range(
                1,
                num_frames
            ):

                # Previous frames
                labels[
                    'x_lag_{}'.format(i)
                ] = labels[
                    'x-coordinate'
                ].shift(i)

                labels[
                    'y_lag_{}'.format(i)
                ] = labels[
                    'y-coordinate'
                ].shift(i)

                # Future frames
                labels[
                    'x_lag_inv_{}'.format(i)
                ] = labels[
                    'x-coordinate'
                ].shift(-i)

                labels[
                    'y_lag_inv_{}'.format(i)
                ] = labels[
                    'y-coordinate'
                ].shift(-i)

                # --------------------------------------------
                # Difference with previous frame
                # --------------------------------------------

                labels[
                    'x_diff_{}'.format(i)
                ] = (
                    labels[
                        'x_lag_{}'.format(i)
                    ]
                    -
                    labels[
                        'x-coordinate'
                    ]
                ).abs()

                labels[
                    'y_diff_{}'.format(i)
                ] = (
                    labels[
                        'y_lag_{}'.format(i)
                    ]
                    -
                    labels[
                        'y-coordinate'
                    ]
                )

                # --------------------------------------------
                # Difference with future frame
                # --------------------------------------------

                labels[
                    'x_diff_inv_{}'.format(i)
                ] = (
                    labels[
                        'x_lag_inv_{}'.format(i)
                    ]
                    -
                    labels[
                        'x-coordinate'
                    ]
                ).abs()

                labels[
                    'y_diff_inv_{}'.format(i)
                ] = (
                    labels[
                        'y_lag_inv_{}'.format(i)
                    ]
                    -
                    labels[
                        'y-coordinate'
                    ]
                )

                # --------------------------------------------
                # Direction/change ratios
                # --------------------------------------------

                labels[
                    'x_div_{}'.format(i)
                ] = (
                    labels[
                        'x_diff_{}'.format(i)
                    ]
                    /
                    (
                        labels[
                            'x_diff_inv_{}'.format(i)
                        ].abs()
                        + eps
                    )
                )

                labels[
                    'y_div_{}'.format(i)
                ] = (
                    labels[
                        'y_diff_{}'.format(i)
                    ]
                    /
                    (
                        labels[
                            'y_diff_inv_{}'.format(i)
                        ].abs()
                        + eps
                    )
                )

            # ------------------------------------------------
            # Remove rows where temporal information is not
            # available
            # ------------------------------------------------

            feature_columns = []

            for i in range(
                1,
                num_frames
            ):

                feature_columns.extend(
                    [
                        'x_lag_{}'.format(i),
                        'x_lag_inv_{}'.format(i),
                        'y_lag_{}'.format(i),
                        'y_lag_inv_{}'.format(i)
                    ]
                )

            labels = labels.dropna(
                subset=feature_columns
            )

            # ------------------------------------------------
            # Current frame must have ball coordinates
            # ------------------------------------------------

            labels = labels.dropna(
                subset=[
                    'x-coordinate',
                    'y-coordinate',
                    'status'
                ]
            )

            # ------------------------------------------------
            # Bounce target
            #
            # Dataset status == 2 means bounce
            # ------------------------------------------------

            labels['target'] = (
                labels['status'] == 2
            ).astype(int)

            labels['status'] = labels[
                'status'
            ].astype(int)

            # ------------------------------------------------
            # Add source information
            # ------------------------------------------------

            labels['game'] = game
            labels['clip'] = clip

            if len(labels) > 0:

                all_data.append(
                    labels
                )

    # --------------------------------------------------------
    # Combine all clips
    # --------------------------------------------------------

    if len(all_data) == 0:

        raise ValueError(
            "No valid training data was found."
        )

    df = pd.concat(
        all_data,
        ignore_index=True
    )

    print()
    print(
        "Total feature rows:",
        len(df)
    )

    print()
    print(
        "Bounce class distribution:"
    )

    print(
        df['target'].value_counts()
    )

    return df


# ============================================================
# CREATE TRAIN / TEST DATA
# ============================================================

def create_train_test(
    df,
    num_frames=3
):

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    colnames_x = (
        [
            'x_diff_{}'.format(i)
            for i in range(
                1,
                num_frames
            )
        ]
        +
        [
            'x_diff_inv_{}'.format(i)
            for i in range(
                1,
                num_frames
            )
        ]
        +
        [
            'x_div_{}'.format(i)
            for i in range(
                1,
                num_frames
            )
        ]
    )

    colnames_y = (
        [
            'y_diff_{}'.format(i)
            for i in range(
                1,
                num_frames
            )
        ]
        +
        [
            'y_diff_inv_{}'.format(i)
            for i in range(
                1,
                num_frames
            )
        ]
        +
        [
            'y_div_{}'.format(i)
            for i in range(
                1,
                num_frames
            )
        ]
    )

    colnames = (
        colnames_x
        +
        colnames_y
    )

    # --------------------------------------------------------
    # Clean invalid values
    # --------------------------------------------------------

    X = df[
        colnames
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = df[
        'target'
    ]

    valid_rows = (
        X.notna().all(axis=1)
        &
        y.notna()
    )

    X = X.loc[
        valid_rows
    ]

    y = y.loc[
        valid_rows
    ]

    print()
    print(
        "Valid rows:",
        len(X)
    )

    # --------------------------------------------------------
    # Stratified split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=5,
        stratify=y
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples :",
        len(X_test)
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        colnames
    )


# ============================================================
# TRAIN CATBOOST BOUNCE MODEL
# ============================================================

def train_bounce_model(
    X_train,
    y_train,
    X_test,
    y_test,
    model_path
):

    print()
    print("=" * 60)
    print("Training CatBoost Bounce Detector")
    print("=" * 60)

    train_pool = ctb.Pool(
        X_train,
        y_train
    )

    test_pool = ctb.Pool(
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # CatBoost classifier
    # --------------------------------------------------------

    model = ctb.CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='F1',
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=50,
        thread_count=-1,
        allow_writing_files=False
    )

    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print()
    print("=" * 60)
    print("Bounce Detection Results")
    print("=" * 60)

    print(
        "TN        :",
        tn
    )

    print(
        "FP        :",
        fp
    )

    print(
        "FN        :",
        fn
    )

    print(
        "TP        :",
        tp
    )

    print(
        "Accuracy  : {:.6f}".format(
            accuracy
        )
    )

    print(
        "Precision : {:.6f}".format(
            precision
        )
    )

    print(
        "Recall    : {:.6f}".format(
            recall
        )
    )

    print(
        "F1 Score  : {:.6f}".format(
            f1
        )
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    output_folder = os.path.dirname(
        model_path
    )

    if output_folder:

        os.makedirs(
            output_folder,
            exist_ok=True
        )

    model.save_model(
        model_path
    )

    print()
    print(
        "Bounce model saved to:"
    )

    print(
        model_path
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train TrackNet bounce detection model"
    )

    parser.add_argument(
        '--path_dataset',
        type=str,
        required=True,
        help='path to TrackNet Dataset folder'
    )

    parser.add_argument(
        '--path_save_model',
        type=str,
        required=True,
        help='output .cbm model path'
    )

    parser.add_argument(
        '--num_frames',
        type=int,
        default=3,
        help='number of temporal frames'
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------

    if args.num_frames < 2:

        raise ValueError(
            "num_frames must be at least 2"
        )

    print("=" * 60)
    print("TrackNet Bounce Training")
    print("=" * 60)

    print(
        "Dataset:",
        args.path_dataset
    )

    print(
        "Output model:",
        args.path_save_model
    )

    print(
        "Feature frames:",
        args.num_frames
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Create features
    # --------------------------------------------------------

    df_features = create_features(
        args.path_dataset,
        args.num_frames
    )

    # --------------------------------------------------------
    # Create train/test sets
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_columns
    ) = create_train_test(
        df_features,
        args.num_frames
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    train_bounce_model(
        X_train,
        y_train,
        X_test,
        y_test,
        args.path_save_model
    )

    print()
    print("=" * 60)
    print("Bounce training completed successfully.")
    print("=" * 60)


if __name__ == '__main__':
    main()