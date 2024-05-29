import torch.nn as nn
import torch

import blocks

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, n_resblock=3, n_feat=32, kernel_size=5, feat_in=False):
        super(UNet, self).__init__()
        print("Creating U-Net")

        InBlock = []
        if not feat_in:
            InBlock.extend([nn.Sequential(
                nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.ReLU(inplace=True)
            )])
        InBlock.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1) for _ in range(n_resblock)])

        # encoder1
        Encoder_1 = [nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )]
        Encoder_1.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                          for _ in range(n_resblock)])
        # encoder2
        Encoder_2 = [nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )]
        Encoder_2.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1)
                          for _ in range(n_resblock)])

        # decoder2
        Decoder_2 = [blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1)
                     for _ in range(n_resblock)]
        Decoder_2.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ))
        # decoder1
        Decoder_1 = [blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                     for _ in range(n_resblock)]
        Decoder_1.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ))

        OutBlock = [blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1) for _ in range(n_resblock)]
        OutBlock.extend([
            nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        ])

        self.inBlock = nn.Sequential(*InBlock)
        self.encoder_1 = nn.Sequential(*Encoder_1)
        self.encoder_2 = nn.Sequential(*Encoder_2)
        self.decoder_2 = nn.Sequential(*Decoder_2)
        self.decoder_1 = nn.Sequential(*Decoder_1)
        self.outBlock = nn.Sequential(*OutBlock)

    def forward(self, x):
        inblock_out = self.inBlock(x)
        encoder_1_out = self.encoder_1(inblock_out)
        encoder_2_out = self.encoder_2(encoder_1_out)
        decoder_2_out = self.decoder_2(encoder_2_out)
        decoder_1_out = self.decoder_1(decoder_2_out + encoder_1_out)
        out = self.outBlock(decoder_1_out + inblock_out)

        mid_loss = None

        return out, mid_loss


class TRANS(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, n_resblock=3, n_feat=32, feat_in=False):
        super(TRANS, self).__init__()
        print("Creating Trans Net")

        self.unet_body = UNet(in_channels=in_channels, out_channels=out_channels,
                                   n_resblock=n_resblock, n_feat=n_feat, feat_in=feat_in)
        self.sigmoid = nn.Sigmoid()

    def clamp_trans(self, trans):
        # trans<0 => 1e-8
        mask = (trans.detach() > 0).float()
        little = torch.ones_like(trans) * 1e-8
        trans = trans * mask + little * (1 - mask)
        # trans>1 => (1 - 1e-8)
        mask = (trans.detach() < 1).float()
        large = torch.ones_like(trans) * (1 - 1e-8)
        trans = trans * mask + large * (1 - mask)
        return trans

    def forward(self, x):
        unet_out, _ = self.unet_body(x)
        est_trans = self.sigmoid(unet_out)
        clamped_est_trans = self.clamp_trans(est_trans)

        mid_loss = None

        return clamped_est_trans, mid_loss

class Smooth_Loss(nn.Module):
    def __init__(self):
        super(Smooth_Loss, self).__init__()

    def forward(self, x):
        loss_smooth = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
                      torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

        return loss_smooth


class PRE_DEHAZE_T(nn.Module):

    def __init__(self, img_channels=3, t_channels=1, n_resblock=3, n_feat=32, device='cuda'):
        super(PRE_DEHAZE_T, self).__init__()
        print("Creating Pre-Dehaze-T Net")
        self.device = device

        self.trans_net = TRANS(in_channels=img_channels, out_channels=t_channels,
                                     n_resblock=n_resblock, n_feat=n_feat)
        self.smooth_loss = Smooth_Loss()

    def forward(self, x):
        b, c, h, w = x.size()

        trans, _ = self.trans_net(x)
        air = torch.ones(b, 1, h, w).to(self.device)

        output = (1 / trans) * x + ((trans - 1) / trans) * air

        mid_loss = self.smooth_loss(trans)

        return output, trans, air, mid_loss


class FusionModule(nn.Module):
    def __init__(self, n_feat, kernel_size=5):
        super(FusionModule, self).__init__()
        print("Creating BRB-Fusion-Module")
        self.block1 = blocks.BinResBlock(n_feat, kernel_size=kernel_size)
        self.block2 = blocks.BinResBlock(n_feat, kernel_size=kernel_size)

    def forward(self, x, y):
        H_0 = x + y

        x_1, y_1, H_1 = self.block1(x, y, H_0)
        x_2, y_2, H_2 = self.block2(x_1, y_1, H_1)

        return H_2


class DEHAZE_T(nn.Module):

    def __init__(self, img_channels=3, t_channels=1, n_resblock=3, n_feat=32, device='cuda'):
        super(DEHAZE_T, self).__init__()
        print("Creating Dehaze-T Net")
        self.device = device

        self.extra_feat = nn.Sequential(
            nn.Conv2d(img_channels, n_feat, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            blocks.ResBlock(n_feat, n_feat, kernel_size=5, stride=1)
        )
        self.fusion_feat = FusionModule(n_feat=n_feat, kernel_size=5)
        self.trans_net = TRANS(in_channels=1, out_channels=t_channels,
                                     n_resblock=n_resblock, n_feat=n_feat, feat_in=True)
        self.smooth_loss = Smooth_Loss()

    def forward(self, x, pre_est_J):
        b, c, h, w = x.size()

        x_feat = self.extra_feat(x)
        pre_est_J_feat = self.extra_feat(pre_est_J)

        fusioned_feat = self.fusion_feat(x_feat, pre_est_J_feat)

        trans, _ = self.trans_net(fusioned_feat)
        air = torch.ones(b, 1, h, w).to(self.device)

        output = (1 / trans) * x + ((trans - 1) / trans) * air

        mid_loss = self.smooth_loss(trans)

        return output, trans, air, mid_loss


class DEHAZE_SGID_PFF(nn.Module):

    def __init__(self, img_channels=3, t_channels=1, n_resblock=3, n_feat=32,
                 pretrain_pre_dehaze_pt='.', device='cuda'):
        super(DEHAZE_SGID_PFF, self).__init__()
        print("Creating Dehaze-SGID-PFF Net")
        self.device = device

        self.pre_dehaze = PRE_DEHAZE_T(img_channels=img_channels, t_channels=t_channels, n_resblock=n_resblock,
                                       n_feat=n_feat, device=device)
        self.dehaze = DEHAZE_T(img_channels=img_channels, t_channels=t_channels, n_resblock=n_resblock,
                               n_feat=n_feat, device=device)

        if pretrain_pre_dehaze_pt != '.':
            self.pre_dehaze.load_state_dict(torch.load(pretrain_pre_dehaze_pt))
            print('Loading pre dehaze model from {}'.format(pretrain_pre_dehaze_pt))

    def forward(self, x):
        pre_est_J, _, _, _ = self.pre_dehaze(x)

        output, trans, air, mid_loss = self.dehaze(x, pre_est_J)

        return pre_est_J, output, trans, air, mid_loss
