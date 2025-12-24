import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from DEANET import DEANet, DEA_Decoder
import torch.nn as nn
import torch.optim as optim
from modules import CGAFusion
import numpy as np
from tqdm import tqdm
from tmc import TMC
from torch.autograd import Variable
import faulthandler
from index import getscore
import pandas as pd
from createmiss import create_miss

faulthandler.enable()


class CustomImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_filenames = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx, miss=False):
        img_path = self.image_filenames[idx]
        image = Image.open(img_path).convert('L')
        # print(type(image))


        if self.transform:
            image = self.transform(image)

        label = self.folder_path.split('/')[-1]  # Simplified label: using the folder name
        return image, label

# Define transformations for your images
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to fit your model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalization for pre-trained models
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
train_epoch = 500
test_epoch = 500
TMC_epoch = 1
optim_step = 20
optim_gamma = 0.5


D_alpha = 1
com_mse_alpha = 1.5
target_mse_alpha = 2
f_alpha = 0.5
E_alpha = 0.1
G_lr = 2e-6
TMC_lr = 1e-5
weight_decay = 0


real_label = torch.ones(batch_size, 1).to(device)
real_label = Variable(real_label.long().cuda())
fake_label = torch.zeros(batch_size, 1).to(device)
fake_label_copy = fake_label.float()
fake_label = Variable(fake_label.long().cuda())





# Paths to your folders
train_paths = ['/mnt/cgshare/DMT/BraTS2020_TrainingData/t1', '/mnt/cgshare/DMT/BraTS2020_TrainingData/t1gd', '/mnt/cgshare/DMT/BraTS2020_TrainingData/t2', '/mnt/cgshare/DMT/BraTS2020_TrainingData/flair']
train_datasets = [CustomImageDataset(folder_path, transform) for folder_path in train_paths]
train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) for dataset in train_datasets]
test_paths = ['/mnt/cgshare/DMT/BraTS2020_ValidationData/t1', '/mnt/cgshare/DMT/BraTS2020_ValidationData/t1gd', '/mnt/cgshare/DMT/BraTS2020_ValidationData/t2', '/mnt/cgshare/DMT/BraTS2020_ValidationData/flair']
test_datasets = [CustomImageDataset(folder_path, transform) for folder_path in test_paths]
test_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) for dataset in test_datasets]


MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
bceloss = nn.BCELoss()
KLloss = nn.KLDivLoss(reduction='sum')


DEA_AE = DEANet(base_dim=32)
DEA_AE.to(device)
TMC = TMC(img_hidden_sz=batch_size)
TMC.to(device)
level_1_mix = CGAFusion(dim=128, reduction=8)
level_1_mix.to(device)
level_2_mix = CGAFusion(dim=64, reduction=4)
level_2_mix.to(device)
Generator = DEA_Decoder(base_dim=32)
Generator.to(device)


optimizer_DEA_AE = optim.Adamax(DEA_AE.parameters(), lr=G_lr)
optimizer_level_1_mix = optim.Adamax(level_1_mix.parameters(), lr=G_lr)
optimizer_level_2_mix = optim.Adamax(level_2_mix.parameters(), lr=G_lr)
optimizer_TMC = optim.Adamax(TMC.parameters(), lr=TMC_lr)
optimizer_Generator = optim.Adamax(Generator.parameters(), lr=G_lr)


def train(dataloader, epoch):
    print("train begining")

    Discriminator_loss = np.zeros(epoch)
    Generator_loss = np.zeros(epoch)
    mse_loss_list = np.zeros(epoch)
    E_loss_list = np.zeros(epoch)
    f_loss_list = np.zeros(epoch)
    dis_loss_list = np.zeros(epoch)
    psnr_list = np.zeros(epoch)
    ssim_list = np.zeros(epoch)
    lpips_list = np.zeros(epoch)

    torch.autograd.set_detect_anomaly(True)

    DEA_AE.train()
    Generator.train()
    level_1_mix.train()
    level_2_mix.train()
    TMC.train()

    # TMC.load_state_dict(torch.load('flair_TMC.pt'))
    # DEA_AE.load_state_dict(torch.load('flair_DEA_AE_1_1_0.3.pt'))
    # Generator.load_state_dict(torch.load('flair_DEA_DE_1_1_0.3.pt'))
    # level_1_mix.load_state_dict(torch.load('flair_l1_mix_1_1_0.3.pt'))
    # level_2_mix.load_state_dict(torch.load('flair_l2_mix_1_1_0.3.pt'))

    for num_epoch in tqdm(range(epoch)):

        Dis_loss = 0
        Gen_loss = 0
        ssim_score = 0
        psnr_score = 0
        lpips_score = 0

        for num_batch in tqdm(range(len(dataloader[0])), ncols=50, position=0, leave=True):
            t1, _ = next(iter(dataloader[0]))
            t1gd, _ = next(iter(dataloader[1]))
            t2, _ = next(iter(dataloader[2]))
            flair, _ = next(iter(dataloader[3]))
            t1 = t1.to(device)
            t1gd = t1gd.to(device)
            t2 = t2.to(device)
            flair = flair.to(device)

            optimizer_TMC.zero_grad()
            optimizer_level_1_mix.zero_grad()
            optimizer_level_2_mix.zero_grad()
            optimizer_DEA_AE.zero_grad()
            optimizer_Generator.zero_grad()

            t1_hat, t1_feature1, t1_feature2 = DEA_AE(t1)
            t1gd_hat, t1gd_feature1, t1gd_feature2 = DEA_AE(t1gd)
            t2_hat, t2_feature1, t2_feature2 = DEA_AE(t2)
            flair_hat, flair_feature1, flair_feature2 = DEA_AE(flair)
            two_mod_mix_feature1 = level_1_mix(flair_feature1, t1_feature1)
            level_1_mix_feature = level_1_mix(two_mod_mix_feature1, t2_feature1)
            two_mod_mix_feature2 = level_2_mix(flair_feature2, t1_feature2)
            level_2_mix_feature = level_2_mix(two_mod_mix_feature2, t2_feature2)
            fake_t1gd = Generator(level_1_mix_feature, level_2_mix_feature)
            fake_t1gd_copy = fake_t1gd.detach().clone()


            fake_dis_label, fake_tmc_loss = TMC(fake_t1gd_copy, fake_label, num_epoch)
            real_dis_label, real_tmc_loss = TMC(flair, real_label, num_epoch)

            D_loss = (fake_tmc_loss + real_tmc_loss) / 2
            Dis_loss += D_loss.item()

            optimizer_TMC.zero_grad()
            D_loss.backward()
            optimizer_TMC.step()

            Dis_loss = Dis_loss / TMC_epoch

            real_dis_label_copy = real_dis_label.detach().clone()
            fake_dis_label_copy = fake_dis_label.detach().clone()
            fake_alpha_copy = fake_dis_label.detach().clone()


            fake_S = torch.sum(fake_alpha_copy, dim=1, keepdim=True)
            real_S = torch.sum(real_dis_label_copy, dim=1, keepdim=True)
            fake_P = fake_dis_label_copy / fake_S
            real_P = real_dis_label_copy / real_S


            t1_mse_loss = L1Loss(t1_hat, t1)
            t1gd_mse_loss = L1Loss(t1gd_hat, t1gd)
            t2_mse_loss = L1Loss(t2_hat, t2)
            flair_mse_loss = L1Loss(flair_hat, flair)

            target_mse_loss = L1Loss(fake_t1gd, t1gd)

            com_mse_loss = com_mse_alpha * flair_mse_loss + com_mse_alpha * t1_mse_loss + com_mse_alpha * t1gd_mse_loss + com_mse_alpha * t2_mse_loss
            mse_loss = com_mse_loss + target_mse_alpha * target_mse_loss

            level_1_loss = MSELoss(level_1_mix_feature, t2_feature1)
            level_2_loss = MSELoss(level_2_mix_feature, t2_feature2)
            f_loss = level_1_loss + level_2_loss

            _, fake_dis_label_copy = torch.max(fake_dis_label_copy.data, 1)
            fake_dis_label_copy = fake_dis_label_copy.unsqueeze(1)
            fake_dis_label_copy = fake_dis_label_copy.float()


            dis_loss = bceloss(fake_dis_label_copy, fake_label_copy)
            e_loss = KLloss(fake_P.log(), real_P)

            G_loss = mse_loss + E_alpha * e_loss + D_alpha * dis_loss + f_alpha * f_loss

            Gen_loss += G_loss.item()
            Discriminator_loss[num_epoch] += Dis_loss
            Generator_loss[num_epoch] += G_loss.item()
            mse_loss_list[num_epoch] += mse_loss.item()
            E_loss_list[num_epoch] += e_loss.item()
            f_loss_list[num_epoch] += f_loss.item()
            dis_loss_list[num_epoch] += dis_loss.item()
            psnr, ssim, lpips = getscore(fake_t1gd.detach(), flair)
            ssim_score = ssim_score + ssim
            psnr_score = psnr_score + psnr
            lpips_score = lpips_score + lpips

            optimizer_level_1_mix.zero_grad()
            optimizer_level_2_mix.zero_grad()
            optimizer_DEA_AE.zero_grad()
            optimizer_Generator.zero_grad()
            G_loss.backward()
            optimizer_level_1_mix.step()
            optimizer_level_2_mix.step()
            optimizer_DEA_AE.step()
            optimizer_Generator.step()

        ssim_score = ssim_score / len(dataloader[0])
        psnr_score = psnr_score / len(dataloader[0])
        lpips_score = lpips_score / len(dataloader[0])

        ssim_list[num_epoch] = ssim_score
        psnr_list[num_epoch] = psnr_score
        lpips_list[num_epoch] = lpips_score

        if num_epoch % 5 == 0 and num_epoch != 0:
            print("\n")
            print(f"Epoch {num_epoch + 1}, Dis_loss: {Dis_loss}, Gen_loss: {Gen_loss}")
            print(f"psnr: {psnr_score},ssim: {ssim_score}, lpips: {lpips_score}")
            score_list = {'psnr': psnr_list, 'ssim': ssim_list, 'lpips': lpips_list}
            score_list = pd.DataFrame(score_list)
            score_list.to_csv('flair_score_list_train.csv', index=False)
            loss_list = {'Dis_loss': Discriminator_loss, 'Gen_loss': Generator_loss, 'rec_loss': mse_loss_list, 'E_loss': E_loss_list, 'dis_loss': dis_loss_list, 'f_loss': f_loss_list}
            loss_list = pd.DataFrame(loss_list)
            loss_list.to_csv('flair_loss_list_train.csv', index=False)

        if num_epoch % 50 == 0 and num_epoch != 0:
            torch.save(DEA_AE.state_dict(), f'flair_DEA_AE_1_1_0.3.pt')
            torch.save(Generator.state_dict(), f'flair_DEA_DE_1_1_0.3.pt')
            torch.save(level_1_mix.state_dict(), f'flair_l1_mix_1_1_0.3.pt')
            torch.save(level_2_mix.state_dict(), f'flair_l2_mix_1_1_0.3.pt')
            torch.save(TMC.state_dict(), f'flair_TMC.pt')

    # return u

# Create an instance of the DEANet model

# 定义损失函数和优化器

def test(dataloader, epoch):
    print("test begining")

    Discriminator_loss = np.zeros(epoch)
    Generator_loss = np.zeros(epoch)
    mse_loss_list = np.zeros(epoch)
    E_loss_list = np.zeros(epoch)
    f_loss_list = np.zeros(epoch)
    dis_loss_list = np.zeros(epoch)
    psnr_list = np.zeros(epoch)
    ssim_list = np.zeros(epoch)
    lpips_list = np.zeros(epoch)

    torch.autograd.set_detect_anomaly(True)

    DEA_AE.train()
    Generator.train()
    level_1_mix.train()
    level_2_mix.train()
    TMC.train()

    TMC.load_state_dict(torch.load('flair_TMC.pt'))
    DEA_AE.load_state_dict(torch.load('flair_DEA_AE_1_1_0.3.pt'))
    Generator.load_state_dict(torch.load('flair_DEA_DE_1_1_0.3.pt'))
    level_1_mix.load_state_dict(torch.load('flair_l1_mix_1_1_0.3.pt'))
    level_2_mix.load_state_dict(torch.load('flair_l2_mix_1_1_0.3.pt'))

    for num_epoch in tqdm(range(epoch)):

        Dis_loss = 0
        Gen_loss = 0
        ssim_score = 0
        psnr_score = 0
        lpips_score = 0

        for num_batch in tqdm(range(len(dataloader[0])), ncols=50, position=0, leave=True):
            t1, _ = next(iter(dataloader[0]))
            t1gd, _ = next(iter(dataloader[1]))
            t2, _ = next(iter(dataloader[2]))
            flair, _ = next(iter(dataloader[3]))
            t1 = t1.to(device)
            t1gd = t1gd.to(device)
            t2 = t2.to(device)
            flair = flair.to(device)

            optimizer_TMC.zero_grad()
            optimizer_level_1_mix.zero_grad()
            optimizer_level_2_mix.zero_grad()
            optimizer_DEA_AE.zero_grad()
            optimizer_Generator.zero_grad()

            t1_hat, t1_feature1, t1_feature2 = DEA_AE(t1)
            t1gd_hat, t1gd_feature1, t1gd_feature2 = DEA_AE(t1gd)
            t2_hat, t2_feature1, t2_feature2 = DEA_AE(t2)
            flair_hat, flair_feature1, flair_feature2 = DEA_AE(flair)


            two_mod_mix_feature1 = level_1_mix(flair_feature1, t1_feature1)
            level_1_mix_feature = level_1_mix(two_mod_mix_feature1, t2_feature1)
            two_mod_mix_feature2 = level_2_mix(flair_feature2, t1_feature2)
            level_2_mix_feature = level_2_mix(two_mod_mix_feature2, t2_feature2)


            fake_t1gd = Generator(level_1_mix_feature, level_2_mix_feature)
            fake_t1gd_copy = fake_t1gd.detach().clone()

            fake_dis_label, fake_tmc_loss = TMC(fake_t1gd_copy, fake_label, num_epoch)
            real_dis_label, real_tmc_loss = TMC(flair, real_label, num_epoch)

            D_loss = (fake_tmc_loss + real_tmc_loss) / 2
            Dis_loss += D_loss.item()

            optimizer_TMC.zero_grad()
            D_loss.backward()
            optimizer_TMC.step()

            Dis_loss = Dis_loss / TMC_epoch

            real_dis_label_copy = real_dis_label.detach().clone()
            fake_dis_label_copy = fake_dis_label.detach().clone()
            fake_alpha_copy = fake_dis_label.detach().clone()

            fake_S = torch.sum(fake_alpha_copy, dim=1, keepdim=True)
            real_S = torch.sum(real_dis_label_copy, dim=1, keepdim=True)
            fake_P = fake_dis_label_copy / fake_S
            real_P = real_dis_label_copy / real_S

            t1_mse_loss = L1Loss(t1_hat, t1)
            t1gd_mse_loss = L1Loss(t1gd_hat, t1gd)
            t2_mse_loss = L1Loss(t2_hat, t2)
            flair_mse_loss = L1Loss(flair_hat, flair)

            target_mse_loss = L1Loss(fake_t1gd, t1gd)

            com_mse_loss = com_mse_alpha * flair_mse_loss + com_mse_alpha * t1_mse_loss + com_mse_alpha * t1gd_mse_loss + com_mse_alpha * t2_mse_loss
            mse_loss = com_mse_loss + target_mse_alpha * target_mse_loss

            level_1_loss = MSELoss(level_1_mix_feature, t1gd_feature1)
            level_2_loss = MSELoss(level_2_mix_feature, t1gd_feature2)
            f_loss = level_1_loss + level_2_loss

            _, fake_dis_label_copy = torch.max(fake_dis_label_copy.data, 1)
            fake_dis_label_copy = fake_dis_label_copy.unsqueeze(1)
            fake_dis_label_copy = fake_dis_label_copy.float()

            dis_loss = bceloss(fake_dis_label_copy, fake_label_copy)
            e_loss = KLloss(fake_P.log(), real_P)

            G_loss = mse_loss + E_alpha * e_loss + D_alpha * dis_loss + f_alpha * f_loss

            Gen_loss += G_loss.item()
            Discriminator_loss[num_epoch] += Dis_loss
            Generator_loss[num_epoch] += G_loss.item()
            mse_loss_list[num_epoch] += mse_loss.item()
            E_loss_list[num_epoch] += e_loss.item()
            f_loss_list[num_epoch] += f_loss.item()
            dis_loss_list[num_epoch] += dis_loss.item()
            psnr, ssim, lpips = getscore(fake_t1gd.detach(), flair)
            ssim_score = ssim_score + ssim
            psnr_score = psnr_score + psnr
            lpips_score = lpips_score + lpips

            optimizer_level_1_mix.zero_grad()
            optimizer_level_2_mix.zero_grad()
            optimizer_DEA_AE.zero_grad()
            optimizer_Generator.zero_grad()
            G_loss.backward()
            optimizer_level_1_mix.step()
            optimizer_level_2_mix.step()
            optimizer_DEA_AE.step()
            optimizer_Generator.step()

        ssim_score = ssim_score / len(dataloader[0])
        psnr_score = psnr_score / len(dataloader[0])
        lpips_score = lpips_score / len(dataloader[0])

        ssim_list[num_epoch] = ssim_score
        psnr_list[num_epoch] = psnr_score
        lpips_list[num_epoch] = lpips_score


        if num_epoch % 5 == 0 and num_epoch != 0:
            print("\n")
            print(f"Epoch {num_epoch + 1}, Dis_loss: {Dis_loss}, Gen_loss: {Gen_loss}")
            print(f"psnr: {psnr_score},ssim: {ssim_score}, lpips: {lpips_score}")
            score_list = {'psnr': psnr_list, 'ssim': ssim_list, 'lpips': lpips_list}
            score_list = pd.DataFrame(score_list)
            score_list.to_csv('flair_score_list_test.csv', index=False)
            loss_list = {'Dis_loss': Discriminator_loss, 'Gen_loss': Generator_loss, 'rec_loss': mse_loss_list, 'E_loss': E_loss_list, 'dis_loss': dis_loss_list, 'f_loss': f_loss_list}
            loss_list = pd.DataFrame(loss_list)
            loss_list.to_csv('flair_loss_list_test.csv', index=False)



# Extract features for each dataset
if __name__ == '__main__':
    train(train_loaders, train_epoch)
    test(test_loaders, test_epoch)


