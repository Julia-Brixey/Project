import os
import torch
import nibabel as nib
from torch.utils.data import Dataset


class LoadDataset(Dataset):
    def __init__(self, directory):
        self.root_dir = directory
        self.images = []
        self.labels = []

        print("Loading Data...")
        for foldername in os.listdir(directory):
            folderpath = os.path.join(directory, foldername)
            for subject in os.listdir(folderpath):
                filepath = os.path.join(folderpath, subject)
                filepath = os.path.join(filepath, "t1", "defaced_mprage.nii")
                if filepath.endswith(".nii"):
                    self.images.append(filepath)
                    self.labels.append(int(foldername))
        print("Data load complete!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        width_min, width_max = 35, 160
        height_min, height_max = 70, 220
        depth_min, depth_max = 30, 210
        image = nib.load(self.images[index])
        image = image.get_fdata()[width_min:width_max, height_min:height_max, depth_min:depth_max]
        image = torch.Tensor(image).unsqueeze(0)
        label = self.labels[index]
        return image, label
