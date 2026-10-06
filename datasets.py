from torch.utils.data import Dataset
import os
import pandas as pd
import cv2
import math
import numpy as np


class trackNetDataset(Dataset):

    def __init__(
        self,
        mode,
        input_height=360,
        input_width=640
    ):

        # ---------------------------------------------------------
        # Dataset locations
        # ---------------------------------------------------------
        self.image_root = './Dataset'
        self.gt_root = './GT_Output'

        # CSV files are inside GT_Output
        self.path_dataset = self.gt_root

        # ---------------------------------------------------------
        # Check mode
        # ---------------------------------------------------------
        assert mode in ['train', 'val'], 'incorrect mode'

        # ---------------------------------------------------------
        # Load labels
        # ---------------------------------------------------------
        label_file = os.path.join(
            self.path_dataset,
            'labels_{}.csv'.format(mode)
        )

        if not os.path.exists(label_file):
            raise FileNotFoundError(
                'Label file not found: {}'.format(label_file)
            )

        self.data = pd.read_csv(label_file)

        print(
            'mode = {}, samples = {}'.format(
                mode,
                self.data.shape[0]
            )
        )

        # ---------------------------------------------------------
        # Image size
        # ---------------------------------------------------------
        self.height = input_height
        self.width = input_width

    # -------------------------------------------------------------
    # Dataset length
    # -------------------------------------------------------------
    def __len__(self):
        return self.data.shape[0]

    # -------------------------------------------------------------
    # Get one sample
    # -------------------------------------------------------------
    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        path = row['path1']
        path_prev = row['path2']
        path_preprev = row['path3']
        path_gt = row['gt_path']

        x = row['x-coordinate']
        y = row['y-coordinate']
        vis = row['visibility']

        # ---------------------------------------------------------
        # Convert NaN coordinates
        # ---------------------------------------------------------
        if pd.isna(x):
            x = -1

        if pd.isna(y):
            y = -1

        # ---------------------------------------------------------
        # Image paths
        #
        # Images are stored in:
        # Dataset/gameX/ClipX/
        # ---------------------------------------------------------
        path = os.path.join(
            self.image_root,
            path
        )

        path_prev = os.path.join(
            self.image_root,
            path_prev
        )

        path_preprev = os.path.join(
            self.image_root,
            path_preprev
        )

        # ---------------------------------------------------------
        # GT path
        #
        # GT images are stored in:
        # GT_Output/gts/gameX/ClipX/
        # ---------------------------------------------------------
        path_gt = os.path.join(
            self.gt_root,
            path_gt
        )

        # ---------------------------------------------------------
        # Load input images
        # ---------------------------------------------------------
        inputs = self.get_input(
            path,
            path_prev,
            path_preprev
        )

        # ---------------------------------------------------------
        # Load ground truth
        # ---------------------------------------------------------
        outputs = self.get_output(path_gt)

        return (
            inputs,
            outputs,
            x,
            y,
            vis
        )

    # -------------------------------------------------------------
    # Load GT heatmap
    # -------------------------------------------------------------
    def get_output(self, path_gt):

        img = cv2.imread(
            path_gt,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            raise FileNotFoundError(
                'GT image not found: {}'.format(path_gt)
            )

        # Resize to TrackNet resolution
        img = cv2.resize(
            img,
            (self.width, self.height)
        )

        # Flatten
        img = np.reshape(
            img,
            (self.width * self.height)
        )

        return img

    # -------------------------------------------------------------
    # Load three consecutive frames
    # -------------------------------------------------------------
    def get_input(
        self,
        path,
        path_prev,
        path_preprev
    ):

        # ---------------------------------------------------------
        # Current frame
        # ---------------------------------------------------------
        img = cv2.imread(path)

        if img is None:
            raise FileNotFoundError(
                'Image not found: {}'.format(path)
            )

        # ---------------------------------------------------------
        # Previous frame
        # ---------------------------------------------------------
        img_prev = cv2.imread(path_prev)

        if img_prev is None:
            raise FileNotFoundError(
                'Previous image not found: {}'.format(path_prev)
            )

        # ---------------------------------------------------------
        # Two frames before
        # ---------------------------------------------------------
        img_preprev = cv2.imread(path_preprev)

        if img_preprev is None:
            raise FileNotFoundError(
                'Previous-previous image not found: {}'.format(
                    path_preprev
                )
            )

        # ---------------------------------------------------------
        # Resize
        # ---------------------------------------------------------
        img = cv2.resize(
            img,
            (self.width, self.height)
        )

        img_prev = cv2.resize(
            img_prev,
            (self.width, self.height)
        )

        img_preprev = cv2.resize(
            img_preprev,
            (self.width, self.height)
        )

        # ---------------------------------------------------------
        # Combine 3 RGB/BGR images
        #
        # 3 channels × 3 frames = 9 channels
        # ---------------------------------------------------------
        imgs = np.concatenate(
            (
                img,
                img_prev,
                img_preprev
            ),
            axis=2
        )

        # ---------------------------------------------------------
        # Normalize
        # ---------------------------------------------------------
        imgs = imgs.astype(
            np.float32
        ) / 255.0

        # ---------------------------------------------------------
        # HWC → CHW
        # ---------------------------------------------------------
        imgs = np.rollaxis(
            imgs,
            2,
            0
        )

        return imgs