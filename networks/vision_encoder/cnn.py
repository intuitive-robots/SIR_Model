import torch
from torch import nn
from torchvision import models
from einops import einops

class Multi_CNN(nn.Module):
    def __init__(self, model_name, modalities, output_dim, use_film, pretrained) -> None:
        super(Multi_CNN, self).__init__()
        
        self.use_film = use_film
        
        self.models = nn.ModuleDict()
        
        for mod in modalities:
            if use_film:
                self.models[mod] = FiLMCNN(model_name, output_dim, pretrained)
            else:
                self.models[mod] = CNN(model_name, output_dim, pretrained)
                
    def forward(self, input, emb=None):
        for key in input:
            if self.use_film:
                input[key] = self.models[key](input[key], emb)
            else:
                input[key] = self.models[key](input[key])
        return input

class CNN(nn.Module):
    def __init__(self, model_name, obs_dim, pretrained, in_channels=3) -> None:
        super(CNN, self).__init__()
        if model_name == "ResNet18":
            self.vision_model = models.resnet18(pretrained=pretrained)
        elif model_name == "ResNet34":
            self.vision_model = models.resnet34(pretrained=pretrained)
        elif model_name == "ResNet50":
            self.vision_model = models.resnet50(pretrained=pretrained)
        elif model_name == "ResNet101":
            self.vision_model = models.resnet101(pretrained=pretrained)
        elif model_name == "ResNet152":
            self.vision_model = models.resnet152(pretrained=pretrained)
        elif model_name == "EffNetB0":
            self.vision_model = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == "ViT":
            self.vision_model = models.vit_b_16(pretrained=pretrained)
        elif model_name == "ConvNext":
            self.vision_model = models.convnext_base(pretrained=pretrained)

        if model_name == "EffNetB0" or model_name == "ConvNext":
            n_inputs = self.vision_model.classifier[-1].in_features
            self.fc_layer = nn.Linear(n_inputs, obs_dim)
            self.vision_model.classifier[-1] = self.fc_layer
        elif model_name == "ViT":
            n_inputs = self.vision_model.heads.head.in_features
            self.fc_layer = nn.Linear(n_inputs, obs_dim)
            self.vision_model.heads.head = self.fc_layer
        else:
            n_inputs = self.vision_model.fc.in_features    
            self.fc_layer = nn.Linear(n_inputs, obs_dim)
            self.vision_model.fc = self.fc_layer
        
        if in_channels != 3:
            self.vision_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
         
    def forward(self, input):
        output = self.vision_model(input)
        return output

class FiLMLayer(nn.Module):
    """
    Uses Feature-wIse Linear Modulation to language condition a conv net
    """
    def __init__(
        self,
        lang_emb_dim,
        channels,
    ):
        super(FiLMLayer, self).__init__()
        # Linear layer with half outputs for beta and half for gamma
        # Consider initializing to 0?
        self.lang_proj = nn.Linear(lang_emb_dim, channels * 2)
        self.relu = nn.ReLU()

    def forward(self, x, lang_emb):
        B, C, H, W = x.shape
        beta, gamma = torch.split(
            self.lang_proj(lang_emb).reshape(B, C * 2, 1, 1), [C, C], 1
        )
        # The FiLM paper suggests modulating by 1 + dGamma instead of just gamma to avoid zeroing activations
        x = (1 + gamma) * x + beta
        return self.relu(x)

class FiLMCNN(nn.Module):
    """
    A ResNet18 block that can be used to process input images and uses FiLM for language conditioning.
    """
    def __init__(
        self,
        model_name,
        obs_dim,
        pretrained,
        lang_emb_dim=1024, # assume the CLIP embedding dimension by default
    ):
        """

        """
        super(FiLMCNN, self).__init__()
        net = CNN(model_name, obs_dim, pretrained)

        # Split up resnet into parts
        layers = nn.ModuleList(net.vision_model.children())
        base_block = []
        conv_blocks = []
        for layer in layers:
            if isinstance(layer, nn.Sequential):
                for sub_layer in layer:
                    conv_blocks.append(sub_layer)
            elif len(conv_blocks) == 0:
                base_block.append(layer)

        self._base_block = nn.Sequential(*base_block)
        self._conv_blocks = nn.ModuleList(conv_blocks)

        film_layers = []
        current_channels = self._base_block(torch.rand((1, 3, 3, 3))).shape[1]
        for conv in conv_blocks:
            current_channels = conv(torch.rand((1, current_channels, 3, 3))).shape[1]
            film_layers.append(FiLMLayer(lang_emb_dim, current_channels))

        self._film_layers = nn.ModuleList(film_layers)
        self._output_channels = current_channels
        
        self.avg_pooling = net.vision_model.avgpool
        self.fc_layer = net.fc_layer

    def forward(self, x, emb):
        org_len = len(x.shape)
        ws = x.shape[1]
        
        if org_len == 2:
            re_input = einops.rearrange(x, "bs (c w h) -> bs c w h", c=3, w=128, h=128)
        else:
            re_input = einops.rearrange(x, "bs ws (c w h) -> (bs ws) c w h", c=3, w=128, h=128)
        
        x = self._base_block(re_input)
        for conv, film in zip(self._conv_blocks, self._film_layers):
            x = conv(x)
            x = film(x, emb)
        
        pool = self.avg_pooling(x)
        flat = torch.flatten(pool, 1)
        fc = self.fc_layer(flat)
        
        if org_len == 2:
            output = fc
        else:
            output = einops.rearrange(fc, "(bs ws) o -> bs ws o", ws=ws)
        
        return output

#######################################################################################################################
############### Following classes are used to pre-train the Cropped-Image-Feature Encoder #############################
#######################################################################################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SimpleImageEncoder(nn.Module):
    def __init__(self, in_channels, embedding_size, device="cpu"):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2), 
            ResidualBlock(256, 512, stride=2)
        )
        # Use adaptive pooling to handle varying input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_size)

        self = self.to(device)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SimpleImageDecoder(nn.Module):
    def __init__(self, embedding_size, img_shape=(3, 128, 128)):
        super().__init__()
        self.start_h, self.start_w = int(img_shape[1]/32), int(img_shape[2]/32)
        self.fc = nn.Linear(embedding_size, 2048*self.start_h*self.start_w)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_shape[0], 3, stride=1, padding=1),
            # nn.ConvTranspose2d(128, img_shape[0], 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.img_shape = img_shape

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.deconv[0].in_channels, self.start_h, self.start_w)
        x = self.deconv(x)
        return x