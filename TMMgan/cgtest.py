import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from DEANET import DEANet
import torch.nn as nn
import torch.optim as optim
from modules import DEABlockTrain, DEBlockTrain, CGAFusion, ImageEncoder
import numpy as np
from tqdm import tqdm
from  tmc import TMC
from torch.autograd import Variable
from copy import deepcopy
import faulthandler
faulthandler.enable()


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.image_filenames[idx]
        image = Image.open(img_path).convert('RGB')  # Assuming RGB images
        if self.transform:
            image = self.transform(image)
        label = self.folder_path.split('/')[-1]  # Simplified label: using the folder name
        return image, label

# Define transformations for your images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit your model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for pre-trained models
])
batch_size = 8
model = DEANet(base_dim=32,tt=False)
TMC = TMC()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
TMC.to(device)
f= torch.tensor([1, 2, 3, 4, 5])
f = f.to(device)

allmix = CGAFusion(dim=128, reduction=8)
optim_step = 20
optim_gamma = 0.5
lr = 1e-4
allmix.to(device)
weight_decay = 0
optimizer_model = torch.optim.Adam(
        model.parameters(), lr=lr)
optimizer_allmix= torch.optim.Adam(
        allmix.parameters(), lr=lr)
optimizer_TMC = torch.optim.Adam(TMC.parameters(), lr=lr)
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
real_label = torch.tensor(np.ones((batch_size, 1), dtype=np.float32))

real_label = Variable(real_label.long().cuda())

fake_label = torch.tensor(np.zeros((batch_size, 1), dtype=np.float32))
fake_label = Variable(fake_label.long().cuda())


    # scheduler1 = torch.optim.lr_scheduler.StepLR( optimizer_model, step_size=optim_step, gamma=optim_gamma)
    # scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_allmix, step_size=optim_step, gamma=optim_gamma)
    # scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer_tmc, step_size=optim_step, gamma=optim_gamma)
# Paths to your folders
folder_paths = ['D:\DMT\\new repainting\BraTS2020_TrainingData\\t1', 'D:\DMT\\new repainting\BraTS2020_TrainingData\\t2', 'D:\DMT\\new repainting\BraTS2020_TrainingData\\flair']
datasets = [CustomImageDataset(folder_path, transform) for folder_path in folder_paths]
dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) for dataset in datasets]
def zxtrain(dataloader, feature, epoch):
    model.train()
    allmix.train()
    TMC.train()

    Dis_loss = 0
    Gen_loss = 0

    for i, ((t1, _), (t2, _), (flair, _)) in enumerate(zip(dataloader[0], dataloader[1], dataloader[2])):
        t1 = t1.to(device)
        t2 = t2.to(device)
        flair = flair.to(device)

        optimizer_TMC.zero_grad()
        optimizer_allmix.zero_grad()
        optimizer_model.zero_grad()

        t1_hat, t1_feature = model(t1, f)
        t2_hat, t2_feature = model(t2, f)
        mix_feature = allmix(t1_feature, t2_feature)

        flair_hat, y = model(flair, mix_feature)

        fake_dis_label, fake_tmc_loss, fake_evidence = TMC(flair_hat, fake_label, epoch)
        real_dis_label, real_tmc_loss, real_evidence = TMC(flair, real_label, epoch)
        D_loss = (fake_tmc_loss + real_tmc_loss) / 2

        D_loss.backward(retain_graph=True)  # Retain graph for subsequent backward pass
        optimizer_TMC.step()

        t1_mse_loss = MSELoss(t1_hat, t1)
        t2_mse_loss = MSELoss(t2_hat, t2)
        e_loss = L1Loss(fake_evidence, real_evidence)
        flair_mse_loss = MSELoss(flair_hat, flair)
        mse_loss = t1_mse_loss + t2_mse_loss + flair_mse_loss

        G_loss = mse_loss + e_loss

        G_loss.backward()
        optimizer_allmix.step()
        optimizer_model.step()

        Dis_loss += D_loss.item()
        Gen_loss += G_loss.item()

    print(f"Epoch {epoch}, Dis_loss: {Dis_loss}, Gen_loss: {Gen_loss}")


# Create an instance of the DEANet model
epoch =10
# 定义损失函数和优化器


def train():

    for i in tqdm(range(epoch)):
        if i%1==0:
            print("epoch:",i)
        torch.multiprocessing.freeze_support()

        t1_hat= zxtrain( dataloaders, 1, i)
        # print(t1_hat.shape)

# Extract features for each dataset
if __name__ == '__main__':
    train()


