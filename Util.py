import os
import torch
import nibabel as nib
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom


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
        print(self.images)
        print(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = nib.load(self.images[index])
        image = image.get_fdata()
        resize_data = (98 / image.shape[0], 128 / image.shape[1], 128 / image.shape[2])
        new_image = zoom(image, resize_data, mode='nearest')
        new_image = torch.Tensor(new_image).unsqueeze(0)
        label = self.labels[index]
        return new_image, label
