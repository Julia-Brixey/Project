import numpy as np
import torch
import nibabel as nib


class NormalizeIntensity(object):
    def __call__(self, image):
        """
        Normalizes the intensity values of a 3D NIfTI image to have zero mean and unit variance.
        """
        img = nib.load(image)
        data = img.get_fdata()
        mean = np.mean(data)
        std = np.std(data)
        data_norm = (data - mean) / std
        img_norm = nib.Nifti1Image(data_norm, img.affine, img.header)
        return torch.from_numpy(img_norm.get_fdata()).float()
