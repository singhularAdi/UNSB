import os.path
import h5py
import random
import torch
from data.base_dataset import BaseDataset
import numpy as np
import cv2


class UnalignedH5Dataset(BaseDataset):
    """
    This dataset class now loads paired data for pansharpening and implements
    the augmentation and preprocessing logic from the PanFeeder reference.
    It returns data in the original A/B dictionary format.
    'lpan' functionality has been removed.
    """

    def __init__(self, opt):
        """Initialize this dataset class."""
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.isTrain = opt.isTrain

        # --- Set max pixel value based on dataset name in path ---
        if "wv3" in opt.dataroot or "qb" in opt.dataroot or "wv2" in opt.dataroot:
            self.max_pixel = 2047.0
        elif "gf2" in opt.dataroot:
            self.max_pixel = 1023.0
        else:
            self.max_pixel = 2047.0
            print(f"Warning: Dataset type not detected in path '{opt.dataroot}'. Defaulting max_pixel to {self.max_pixel}.")

        dataroot = None
        if opt.isTrain == True:
            dataroot = opt.dataroot
            print(f"Loading data from H5 file: {opt.dataroot}")
        else:
            dataroot = opt.val_dataroot
            print(f"Loading data from H5 file: {opt.val_dataroot}")
        if dataroot is None:
            dataroot = opt.dataroot
        with h5py.File(dataroot, 'r') as f:
            # Transpose from (N, C, H, W) to (N, H, W, C) for numpy augmentations
            self.pan_data = f['pan'][:].transpose(0, 2, 3, 1)
            self.lms_data = f['lms'][:].transpose(0, 2, 3, 1)
            self.ms_data = f['ms'][:].transpose(0, 2, 3, 1)


            self.has_gt = 'gt' in f
            if self.has_gt:
                self.gt_data = f['gt'][:].transpose(0, 2, 3, 1)
            else:
                self.gt_data = None

        self.A_size = self.pan_data.shape[0]  # Domain A (PAN)
        self.B_size = self.lms_data.shape[0]  # Domain B (LMS/MS/GT)
        print(f"Found {self.A_size} images for domain A and {self.B_size} images for domain B.")


    def _augment(self, pan, lms, ms, gt=None):
        """
        Applies random *unpaired* augmentations.
        Crops A (pan) and B (lms, ms, gt) from independent random locations.
        Flips and rotations are already independent.
        """
        # --- 1. Random Crop and Resize (Applied *independently*) ---
        if 'crop' in self.opt.preprocess:
            assert ms.shape[0] == 16  # Assuming this is the base low-res size
            ms_size = ms.shape[0]
            ratio = 1
            ms_p = np.round(ms_size * ratio).astype(int)
            pan_p = 4 * ms_p

            h_pan, w_pan, _ = pan.shape
            pan_x_a = random.randrange(0, w_pan - pan_p + 1)
            pan_y_a = random.randrange(0, h_pan - pan_p + 1)
            pan = pan[pan_y_a:pan_y_a + pan_p, pan_x_a:pan_x_a + pan_p, :]

            h_ms, w_ms, _ = ms.shape
            ms_x_b = random.randrange(0, w_ms - ms_p + 1)
            ms_y_b = random.randrange(0, h_ms - ms_p + 1)
            pan_x_b, pan_y_b = 4 * ms_x_b, 4 * ms_y_b  # Coords for B's high-res images

            ms = ms[ms_y_b:ms_y_b + ms_p, ms_x_b:ms_x_b + ms_p, :]
            lms = lms[pan_y_b:pan_y_b + pan_p, pan_x_b:pan_x_b + pan_p, :]
            if self.has_gt:
                gt = gt[pan_y_b:pan_y_b + pan_p, pan_x_b:pan_x_b + pan_p, :]

            pan = cv2.resize(pan, (4 * ms_size, 4 * ms_size), interpolation=cv2.INTER_CUBIC)
            lms = cv2.resize(lms, (4 * ms_size, 4 * ms_size), interpolation=cv2.INTER_CUBIC)
            ms = cv2.resize(ms, (ms_size, ms_size), interpolation=cv2.INTER_CUBIC)
            if self.has_gt:
                gt = cv2.resize(gt, (4 * ms_size, 4 * ms_size), interpolation=cv2.INTER_CUBIC)

            # Ensure single-channel images have a channel dimension
            if pan.ndim == 2: pan = pan[..., np.newaxis]

        # --- 2. Random Independent Flips (This logic is already unpaired) ---
        if not self.opt.no_flip:
            if random.random() < 0.5: pan = np.fliplr(pan)
            if random.random() < 0.5: lms = np.fliplr(lms)
            if random.random() < 0.5: ms = np.fliplr(ms)
            if self.has_gt and random.random() < 0.5: gt = np.fliplr(gt)

        if not self.opt.no_flip:
            if random.random() < 0.5: pan = np.flipud(pan)
            if random.random() < 0.5: lms = np.flipud(lms)
            if random.random() < 0.5: ms = np.flipud(ms)
            if self.has_gt and random.random() < 0.5: gt = np.flipud(gt)

        # --- 3. Random Independent Rotation (This logic is already unpaired) ---
        if not self.opt.no_rot:
            if random.random() < 0.5:
                pan = np.rot90(pan, k=random.randint(1, 3))
            if random.random() < 0.5:
                lms = np.rot90(lms, k=random.randint(1, 3))
            if random.random() < 0.5:
                ms = np.rot90(ms, k=random.randint(1, 3))
            if self.has_gt and random.random() < 0.5:
                gt = np.rot90(gt, k=random.randint(1, 3))

        return pan, lms, ms, gt

    def _np2tensor(self, np_array):
        """Converts a NumPy array (H, W, C) to a PyTorch tensor (C, H, W) and normalizes to [-1, 1]."""
        if np_array is None:
            return None
        np_array = np.ascontiguousarray(np_array)
        # Transpose from (H, W, C) to (C, H, W)
        tensor = torch.from_numpy(np_array.transpose(2, 0, 1)).float()
        # Normalize to [-1, 1]
        tensor = 2.0 * tensor / self.max_pixel - 1.0
        return tensor

    def __getitem__(self, index):
        """Return a data point and its metadata information."""

        index_A = index % self.A_size
        pan_np = self.pan_data[index_A].copy()

        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)

        lms_np = self.lms_data[index_B].copy()
        ms_np = self.ms_data[index_B].copy()
        gt_np = self.gt_data[index_B].copy() if self.has_gt else None


        # Store original, un-augmented data for testing/validation
        return_dict = {}
        if not self.isTrain:
            return_dict['pan_orig'] = pan_np.transpose(2, 0, 1).astype(np.float32)
            return_dict['lms_orig'] = lms_np.transpose(2, 0, 1).astype(np.float32)
            return_dict['ms_orig'] = ms_np.transpose(2, 0, 1).astype(np.float32)
            if self.has_gt:
                return_dict['gt_orig'] = gt_np.transpose(2, 0, 1).astype(np.float32)

        if self.isTrain:
            pan_np, lms_np, ms_np, gt_np = self._augment(pan_np, lms_np, ms_np, gt_np)

        A_final = self._np2tensor(pan_np) # pan
        B_final = self._np2tensor(lms_np) # lms

        if A_final.shape[0] != B_final.shape[0] and A_final.shape[0] == 1:
             A_final = A_final.expand(B_final.shape[0], -1, -1)

        ms_final = self._np2tensor(ms_np)
        gt_final = self._np2tensor(gt_np)

        # Populate the final dictionary
        return_dict.update({
            'A': A_final, 'B': B_final, 'ms': ms_final,
            'A_paths': f'A_{index_A}', 'B_paths': f'B_{index_B}', 'lms': B_final,
            'pan': A_final
        })
        if self.has_gt:
            return_dict['gt'] = gt_final

        return return_dict

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

