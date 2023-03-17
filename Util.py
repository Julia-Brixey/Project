import os

import numpy as np
import torch
import nibabel as nib
from torch.utils.data import Dataset


class LoadDataset(Dataset):
    def __init__(self, directory):
        self.root_dir = directory
        self.samples = []
        self.labels = []

        for foldername in os.listdir(directory):
            folderpath = os.path.join(directory, foldername)
            for filename in os.listdir(folderpath):
                filepath = os.path.join(folderpath, filename)
                if filepath.endswith(".nii"):
                    self.samples.append(filepath)
                    self.labels.append(int(foldername))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = nib.load(self.samples[index]).get_fdata(dtype=np.float32)
        sample = torch.tensor(sample).unsqueeze(0)
        label = torch.tensor(self.labels[index])
        return sample, label
