import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision

class ImageEncoder(nn.Module):
    def __init__(self,num_image_embeds=1, img_embed_pool_type="avg"):
        super(ImageEncoder, self).__init__()

        self.num_image_embeds=num_image_embeds
        self.img_embed_pool_type=img_embed_pool_type
        self.toRGB = torchvision.transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x)
        model = torchvision.models.resnet18(pretrained=True)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if self.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if self.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((self.num_image_embeds, 1))
        elif self.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif self.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif self.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif self.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.toRGB(x)
        out = self.model(out)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048
# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return torch.mean((A + B))
class TMC(nn.Module):
    def __init__(self,img_hidden_sz=4,num_image_embeds=1,img_embed_pool_type="max",hidden=[ 512],dropout=0.1,n_classes=2):
        super(TMC, self).__init__()

        self.n_classes=n_classes
        # self.tgt=tgt
        # self.i_epoch=i_epoch
        self.dropout = dropout
        self.hidden=hidden
        self.num_image_embeds=num_image_embeds
        self.img_hidden_sz=img_hidden_sz
        self.img_embed_pool_type = img_embed_pool_type
        self.rgbenc = ImageEncoder(num_image_embeds=self.num_image_embeds,img_embed_pool_type=self.img_embed_pool_type)
        self.annealing_epoch=10
        # self.depthenc = ImageEncoder(num_image_embeds=self.num_image_embeds,num_image_embeds_type=self.num_image_embeds_type)
        depth_last_size = self.img_hidden_sz * self.num_image_embeds
        rgb_last_size = self.img_hidden_sz * self.num_image_embeds

        self.clf_rgb = nn.ModuleList()
        rgb_last_size = self.img_hidden_sz * self.num_image_embeds
        for hidden in self.hidden:
            # self.clf_rgb.append(nn.Linear(rgb_last_size, hidden))
            self.clf_rgb.append(nn.Linear(512, hidden))
            self.clf_rgb.append(nn.ReLU())
            self.clf_rgb.append(nn.Dropout(self.dropout))
            rgb_last_size = hidden
        self.clf_rgb.append(nn.Linear(rgb_last_size, self.n_classes))

    def DS_Combin_two(self, alpha1, alpha2):
        # Calculate the merger of two DS evidences
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.args.n_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.args.n_classes, 1), b[1].view(-1, 1, self.args.n_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = self.args.n_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    def forward(self, rgb, tgt, i_epoch):

        rgb = self.rgbenc(rgb)
        rgb = torch.flatten(rgb, start_dim=1)

        # print("rgb shape:")
        # print(rgb.shape)
        rgb_out = rgb
        # print("before zz rgb_out shape:", rgb_out.shape)
        # rgb_out = torch.transpose(rgb_out, 0, 1)###转置
        # print("after zz rgb_out shape:", rgb_out.shape)
        for layer in self.clf_rgb:
            rgb_out = layer(rgb_out)  # 先进行线性变换
            if isinstance(layer, nn.Linear):  # 检查当前层是否是线性层
                rgb_out = F.relu(rgb_out)  # 应用ReLU激活函数
                rgb_out = F.dropout(rgb_out, p=self.dropout)  # 应用dropout

        rgb_evidence = F.softplus(rgb_out)

        rgb_alpha = rgb_evidence+1

        # print("rgb_alpha shape:",rgb_alpha.shape)
        loss = ce_loss(tgt, rgb_alpha, self.n_classes, i_epoch, self.annealing_epoch)

        return rgb_alpha, loss