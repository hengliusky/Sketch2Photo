import torch
import torch.nn as nn
import torch.nn.functional as F
from .ffc import FFCUnit
from .swin_attention import SwinTransformerBlock


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def normal(feat, eps=1e-5):
    feat_mean, feat_std = calc_mean_std(feat, eps)
    normalized = (feat - feat_mean) / feat_std
    return normalized


class SANet(nn.Module):
    def __init__(self, in_dim):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.g = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.h = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, content_feat, style_feat):
        B, C, H, W = content_feat.size()
        F_Fc_norm = self.f(normal(content_feat)).view(B, -1, H * W).permute(0, 2, 1)
        B, C, H, W = style_feat.size()
        G_Fs_norm = self.g(normal(style_feat)).view(B, -1, H * W)
        energy = torch.bmm(F_Fc_norm, G_Fs_norm)
        attention = self.softmax(energy)
        H_Fs = self.h(style_feat).view(B, -1, H * W)
        out = torch.bmm(H_Fs, attention.permute(0, 2, 1))
        B, C, H, W = content_feat.size()
        out = out.view(B, C, H, W)
        out = self.out_conv(out)
        out += content_feat
        return out


class MaxPoolingBlock_conv(nn.Module):
    '''
    SAP block
    '''

    def __init__(self, input_nc,
                 ngf=64,
                 pooling=torch.nn.MaxPool2d,
                 kernel_size=13,
                 padding_type='reflect',
                 norm_layer=None,
                 activation=nn.ReLU(True),
                 use_dropout=False,
                 use_global_pooling=False,  # no use
                 sap_branches=[1, 5, 9, 13],
                 use_class_feat=False):
        super(MaxPoolingBlock_conv, self).__init__()
        self.input_nc = input_nc
        self.use_class_feat = use_class_feat
        self.pooling_nc = len(sap_branches)

        pooling_list = []
        pooling_ngf = ngf // self.pooling_nc
        for pk in sap_branches:
            pooling_list += [[nn.ReflectionPad2d(pk // 2), pooling(pk, stride=1),
                              nn.Conv2d(1, pooling_ngf, kernel_size=kernel_size, padding=kernel_size // 2),
                              activation,
                              nn.Conv2d(pooling_ngf, pooling_ngf, kernel_size=3, padding=1),
                              activation
                              ]]
        for n in range(len(pooling_list)):
            setattr(self, 'pooling_' + str(n), nn.Sequential(*pooling_list[n]))
        atten_input_ch =  1
        attention_block = [nn.Conv2d(atten_input_ch, ngf, kernel_size=kernel_size, padding=kernel_size // 2),
                           # use large kernel size at the beginning
                           activation,
                           nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
                           activation,
                           nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
                           # nn.AdaptiveAvgPool2d((1, 1)),
                           torch.nn.Softmax2d(),
                           ]  # [N, self.pooling_nc, 1, 1]

        self.attention_block = nn.Sequential(*attention_block)

    def forward(self, x):
        # convert input to one channel
        if x.size()[1] != 1:
            x = torch.mean(x, dim=1, keepdim=True)
        # turn 1 as edge, 0 as background
        x_inv = 0. - x

        pooling = getattr(self, 'pooling_' + str(0))
        out = 0. - pooling(x_inv)
        for n in range(1, self.pooling_nc):
            pooling = getattr(self, 'pooling_' + str(n))
            out = torch.cat((out, 0. - pooling(x_inv)), dim=1)

        if self.use_class_feat:
            x = x.repeat(1, 3, 1, 1)
            class_feat = self.pre_classifier(x)
            up_0 = torch.nn.functional.interpolate(class_feat[0], scale_factor=2, mode='bilinear', align_corners=True)
            up_1 = torch.nn.functional.interpolate(class_feat[1], scale_factor=4, mode='bilinear', align_corners=True)
            up_2 = torch.nn.functional.interpolate(class_feat[2], scale_factor=8, mode='bilinear', align_corners=True)
            class_feat_cat = torch.cat((up_0, up_1, up_2), dim=1)
            attention = self.attention_block(class_feat_cat)
        else:
            attention = self.attention_block(x)
        out = out * attention
        return out, attention


class Generator(nn.Module):
    def __init__(self, input_c=3, output_c=3, num_filters=64):
        super().__init__()
        norm = nn.BatchNorm2d
        self.max_pool = MaxPoolingBlock_conv(3,
                                             ngf=64,
                                             kernel_size=13,
                                             padding_type='reflect',
                                             norm_layer=nn.BatchNorm2d,
                                             activation=nn.ReLU(True) ,
                                             use_dropout=False,
                                             use_global_pooling=False,
                                             sap_branches=[1, 5, 9, 13],
                                             use_class_feat=False)
        
        self.downOuterMost = nn.Conv2d(num_filters, num_filters, kernel_size=4,
                                       stride=2, padding=1, bias=False)
        self.model1down = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(num_filters, num_filters * 2, kernel_size=4,
                                                 stride=2, padding=1, bias=False),
              norm(num_filters * 2)])
        self.model2down = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4,
                                                 stride=2, padding=1, bias=False),
              norm(num_filters * 4)])
        self.model3down = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4,
                                                 stride=2, padding=1, bias=False),
              norm(num_filters * 8)])
        self.model4down = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4,
                                                 stride=2, padding=1, bias=False),
              norm(num_filters * 8)])
        self.model5down = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4,
                                                 stride=2, padding=1, bias=False),
              norm(num_filters * 8)])
        self.model6down = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4,
                                                 stride=2, padding=1, bias=False),
              norm(num_filters * 8)])
        self.modelInnerestdown = nn.Sequential(
            *[nn.LeakyReLU(0.2, True), nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4,
                                                 stride=2, padding=1, bias=False), ])
        self.modelInnerestup = nn.Sequential(
            *[nn.ReLU(True), nn.ConvTranspose2d(num_filters * 8, num_filters * 8, kernel_size=4,
                                                stride=2, padding=1, bias=False),
              norm(num_filters * 8)])
        self.model6up = nn.ConvTranspose2d(num_filters * 8 * 2, num_filters * 8, kernel_size=4,
                                           stride=2, padding=1, bias=False)
        self.model5up = nn.ConvTranspose2d(num_filters * 8 * 2, num_filters * 8, kernel_size=4,
                                           stride=2, padding=1, bias=False)
        self.model4up = nn.ConvTranspose2d(num_filters * 8 * 2, num_filters * 8, kernel_size=4,
                                           stride=2, padding=1, bias=False)
        self.model3up = nn.ConvTranspose2d(num_filters * 8 * 2, num_filters * 4, kernel_size=4,
                                           stride=2, padding=1, bias=False)
        self.model2up = nn.ConvTranspose2d(num_filters * 8, num_filters * 2, kernel_size=4,
                                           stride=2, padding=1, bias=False)
        self.model1up = nn.ConvTranspose2d(num_filters * 4, num_filters, kernel_size=4,
                                           stride=2, padding=1, bias=False)

        self.att7 = SwinTransformerBlock(dim=1024, input_resolution=(2, 2),
                                         num_heads=16, window_size=2,
                                         shift_size=0,
                                         mlp_ratio=4,
                                         qkv_bias=True, qk_scale=None,
                                         drop=0, attn_drop=0,
                                         drop_path=0.1,
                                         norm_layer=nn.LayerNorm)

        self.att6 = SwinTransformerBlock(dim=1024, input_resolution=(4, 4),
                                         num_heads=16, window_size=4,
                                         shift_size=0,
                                         mlp_ratio=4,
                                         qkv_bias=True, qk_scale=None,
                                         drop=0, attn_drop=0,
                                         drop_path=0.1,
                                         norm_layer=nn.LayerNorm)

        self.att5 = SwinTransformerBlock(dim=1024, input_resolution=(8, 8),
                                         num_heads=16, window_size=8,
                                         shift_size=0,
                                         mlp_ratio=4,
                                         qkv_bias=True, qk_scale=None,
                                         drop=0, attn_drop=0,
                                         drop_path=0.1,
                                         norm_layer=nn.LayerNorm)

        self.att4 = SwinTransformerBlock(dim=1024, input_resolution=(16, 16),
                                         num_heads=16, window_size=8,
                                         shift_size=0,
                                         mlp_ratio=4,
                                         qkv_bias=True, qk_scale=None,
                                         drop=0, attn_drop=0,
                                         drop_path=0.1,
                                         norm_layer=nn.LayerNorm)

        self.att4_sft = SwinTransformerBlock(dim=1024, input_resolution=(16, 16),
                                             num_heads=16, window_size=8,
                                             shift_size=4,
                                             mlp_ratio=4,
                                             qkv_bias=True, qk_scale=None,
                                             drop=0, attn_drop=0,
                                             drop_path=0.1,
                                             norm_layer=nn.LayerNorm)

        self.att3 = SwinTransformerBlock(dim=512, input_resolution=(32, 32),
                                         num_heads=8, window_size=8,
                                         shift_size=0,
                                         mlp_ratio=4,
                                         qkv_bias=True, qk_scale=None,
                                         drop=0, attn_drop=0,
                                         drop_path=0.1,
                                         norm_layer=nn.LayerNorm)

        self.att3_sft = SwinTransformerBlock(dim=512, input_resolution=(32, 32),
                                             num_heads=8, window_size=8,
                                             shift_size=4,
                                             mlp_ratio=4,
                                             qkv_bias=True, qk_scale=None,
                                             drop=0, attn_drop=0,
                                             drop_path=0.1,
                                             norm_layer=nn.LayerNorm)

        self.att2 = SwinTransformerBlock(dim=256, input_resolution=(64, 64),
                                         num_heads=4, window_size=8,
                                         shift_size=0,
                                         mlp_ratio=4,
                                         qkv_bias=True, qk_scale=None,
                                         drop=0, attn_drop=0,
                                         drop_path=0.1,
                                         norm_layer=nn.LayerNorm)

        self.att2_sft = SwinTransformerBlock(dim=256, input_resolution=(64, 64),
                                             num_heads=4, window_size=8,
                                             shift_size=4,
                                             mlp_ratio=4,
                                             qkv_bias=True, qk_scale=None,
                                             drop=0, attn_drop=0,
                                             drop_path=0.1,
                                             norm_layer=nn.LayerNorm)

        self.att1 = SwinTransformerBlock(dim=128, input_resolution=(128, 128),
                                         num_heads=2, window_size=8,
                                         shift_size=0,
                                         mlp_ratio=4,
                                         qkv_bias=True, qk_scale=None,
                                         drop=0, attn_drop=0,
                                         drop_path=0.1,
                                         norm_layer=nn.LayerNorm)

        self.att1_sft = SwinTransformerBlock(dim=128, input_resolution=(128, 128),
                                             num_heads=2, window_size=8,
                                             shift_size=4,
                                             mlp_ratio=4,
                                             qkv_bias=True, qk_scale=None,
                                             drop=0, attn_drop=0,
                                             drop_path=0.1,
                                             norm_layer=nn.LayerNorm)

        self.upOuterMost = nn.Sequential(*[nn.ReLU(True), nn.ConvTranspose2d(64 * 2, output_c, kernel_size=4,
                                                                             stride=2, padding=1), nn.Tanh()])

        self.ffc_unit1 = FFCUnit(in_channels=64, out_channels=64, kernel_size=5)
        self.ffc_unit2 = FFCUnit(in_channels=128, out_channels=128, kernel_size=5)
        self.ffc_unit3 = FFCUnit(in_channels=256, out_channels=256, kernel_size=5)
        self.ffc_unit4 = FFCUnit(in_channels=512, out_channels=512, kernel_size=5)
        #self.ffc_unit5 = FFCUnit(in_channels=512, out_channels=512, kernel_size=5)

        self.deconv_for_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 1024, 3, stride=2, padding=1, output_padding=1),  # output is 64 * 64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),  # output is 64 * 64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # output is 64 * 64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # output is 128 * 128
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # output is 256 * 256
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # output is 64 * 64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # output is 64 * 64
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1, output_padding=0),  # output is 256 * 256
            nn.Tanh()
        )

    def forward(self, x):
        #sap module
        x, _ = self.max_pool(x)
    
        # sketch encoder
        out1 = self.downOuterMost(x)
        out2 = self.model1down(self.ffc_unit1(out1))
        out3 = self.model2down(self.ffc_unit2(out2))
        out4 = self.model3down(self.ffc_unit3(out3))
        out5 = self.model4down(self.ffc_unit4(out4))
        out6 = self.model5down(out5)
        out7 = self.model6down(out6)


        outmiddown = self.modelInnerestdown(out7)
        outmidup = self.modelInnerestup(outmiddown)

        out7cat = torch.cat([out7, outmidup], dim=1)
        att7 = self.att7(out7cat)
        out7up = self.model6up(att7)

        out6cat = torch.cat([out6, out7up], dim=1)
        att6 = self.att6(out6cat)
        out6up = self.model5up(att6)

        out5cat = torch.cat([out5, out6up], dim=1)
        att5 = self.att5(out5cat)
        out5up = self.model4up(att5)

        out4cat = torch.cat([out4, out5up], dim=1)
        att4 = self.att4(out4cat)
        att4 = self.att4_sft(att4)
        out4up = self.model3up(att4)

        #out3 = self.ffc_unit3(out3)
        out3cat = torch.cat([out3, out4up], dim=1)
        att3 = self.att3(out3cat)
        att3 = self.att3_sft(att3)
        out3up = self.model2up(att3)

        #out2 = self.ffc_unit4(out2)
        out2cat = torch.cat([out2, out3up], dim=1)
        att2 = self.att2(out2cat)
        att2 = self.att2_sft(att2)
        out2up = self.model1up(att2)

        #out1 = self.ffc_unit5(out1)
        out1cat = torch.cat([out1, out2up], dim=1)
        att1 = self.att1(out1cat)
        att1 = self.att1_sft(att1)
        out = self.upOuterMost(att1)
        out_guide = self.deconv_for_decoder(out7)
        return out, out_guide


if __name__ == '__main__':
    g = Generator()
    # print(g)
    num_params = 0
    for p in g.parameters():
        num_params += p.numel()

    print("The number of parameters: {}".format(num_params))
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 3, 256, 256)
    out = g(x)
    print("out.shape ", out[0].shape)
    print(g)
    """
    import cv2
    import matplotlib.pyplot as plt

    img = cv2.imread('1-images.jpg')
    x = torch.Tensor(img)
    x = x.permute(2, 0, 1).unsqueeze(0)
    out = g(x)
    print(out[0].shape, "  ", out[1].shape, "   ", out[2].shape)
    res = out[2]
    for i in range(64):  # 可视化了32通道
        ax = plt.subplot(8, 8, i + 1)
        ax.set_title('Feature {}'.format(i))
        ax.axis('off')
        ax.set_title('new—conv1-image')
        plt.imshow(out[2][0].data.cpu().numpy()[i, :, :], cmap='jet')
    plt.show()
    """
