import torch.nn as nn
import timm
import segmentation_models_pytorch as smp
from utils.config import model_config


class UNetViT(nn.Module):
    def __init__(self, abunet_config=model_config['unetvit']):
        super(UNetViT, self).__init__()

        unet_config = abunet_config['unet']
        vit_config = abunet_config['vit']

        self.unet = smp.Unet(
            encoder_name=unet_config['encoder_name'],
            encoder_weights='imagenet' if unet_config['pretrained'] else None,
            encoder_depth=unet_config['encoder_depth'],
            decoder_channels=unet_config['decoder_channels'],
            decoder_attention_type=unet_config['decoder_attention_type'],
            in_channels=3,
            classes=3,
            activation=None
        )

        if unet_config['transfer']:
            for param in self.unet.parameters():
                param.requires_grad = False

        self.adaptive_pool = nn.AdaptiveAvgPool2d(vit_config['patch_size_to_vit'])

        self.vit = timm.create_model(model_name=vit_config['model_name'],
                                     pretrained=vit_config['pretrained'])
        self.vit.head = nn.Identity()

        if vit_config['transfer']:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.vit_encoder = nn.TransformerEncoderLayer(d_model=768, nhead=12)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=1)
        )

    def forward(self, x):
        pixel_lv = self.adaptive_pool(self.unet(x))
        patch_lv = self.vit_encoder(self.vit(pixel_lv))
        slide_lv = self.classifier(patch_lv)

        return slide_lv


# ## Check
# import torch
# model = UNetViT().to('cuda')
# patch = torch.randn(10, 3, 512, 512).to('cuda')     # [B, C, H, W]
# unet_output = model.unet.forward(patch)
# print(unet_output.size())   # [10, 3, 512, 512]
# output = model(patch)
# print(output.size())    # [10, 1]
