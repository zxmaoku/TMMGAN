import torch
import kornia.losses
import lpips


def getscore(data1, data2):
    psnr_value = kornia.losses.psnr_loss(data1, data2, 2)
    ssim_value = kornia.losses.ssim_loss(data2, data1, window_size=11, reduction='mean')

    # ssim_value1 = cv2.compareSSIM(data1.detach().cpu().numpy(), data2.detach().cpu().numpy(), data_range=data2.detach().cpu().numpy().max() - data2.detach().cpu().numpy().min())
    # win_size = min(data1.shape[0], data1.shape[1]) - 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
    data1 = data1.unsqueeze(0) if data1.dim() == 3 else data1
    data2 = data2.unsqueeze(0) if data2.dim() == 3 else data2
    # 计算LPIPS
    # ssim_value1 = ssim(data1.cpu().numpy(), data2.cpu().numpy(), win_size=7, multichannel=True, channel_axis=2)

    lpips_value = lpips_model(data1, data2).mean().item()

    return -psnr_value, 1 - ssim_value, lpips_value
    # return psnr_value, ssim_value, lpips_value, ssim_value1