import os
import torch
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import cv2 as cv
import setting
import heat_kernel

# Define your dataset class
class mydataset(Dataset):
    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform
    
    def __len__(self):
        return len(self.train_image_file_paths)
    
    def __getitem__(self, idx):
        gif_root = self.train_image_file_paths[idx]
        gif_name = gif_root.split(os.path.sep)[-1]
        gif = cv.VideoCapture(gif_root)
        frames = []
        data = []
        while True:
            ret, frame = gif.read()
            if not ret:
                break
            frames.append(frame)
        gif.release()
       
        if self.transform is not None:
            for i in range(frames.__len__()-1):
                if i % 10 == 0:
                    data.append(self.transform(frames[i]))
        if data is not None:
            data =  torch.stack(data)
        kernel_size = int(gif_name.split('_')[0])
        timestep = int(gif_name.split('_')[1].split('.')[0])
        kernel = heat_kernel.heat_kernel(timestep, kernel_size)
        return data, kernel.data

# Set up transformations for your input images 
transform = transforms.Compose([
    transforms.ToTensor()
])

def get_train_data_loader():
    dataset = mydataset(setting.TRAIN_DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)

# data = get_train_data_loader()
# for d,k in data:
#     print(d.shape)
#     print(k[0].shape)