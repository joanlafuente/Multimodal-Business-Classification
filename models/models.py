import torch.nn as nn
import torchvision
import torch
from einops import rearrange
# Conventional and convolutional neural network

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
class ConTextTransformer(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()

        # Visual feature extractor
        resnet50 = torchvision.models.resnet50(pretrained = True)
        modules=list(resnet50.children())[:-2]
        self.resnet50=nn.Sequential(*modules)
        for param in self.resnet50.parameters():
            param.requires_grad = True
        self.num_cnn_features = 64  # 8x8
        self.dim_cnn_features = 2048
        self.dim_fasttext_features = 300

        # Embeddings for the visual and textual features
        self.cnn_feature_to_embedding = nn.Linear(self.dim_cnn_features, dim)
        self.fasttext_feature_to_embedding = nn.Linear(self.dim_fasttext_features, dim)

        # Learnable position embeddings (for the visual features) and CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_cnn_features + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # The Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        encoder_norm = nn.LayerNorm(dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification Head (MLP)
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.PReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.PReLU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, txt, mask=None):
        x = self.resnet50(img)
        x = rearrange(x, 'b d h w -> b (h w) d') # this makes a sequence of 64 visual features
        x = self.cnn_feature_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x2 = self.fasttext_feature_to_embedding(txt.float())
        x = torch.cat((x,x2), dim=1)
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
    

class Transformer(nn.Module):
    def __init__(self, num_classes, depth_transformer, heads_transformer, dim_fc_transformer):
        super(Transformer, self).__init__()

        full_cnn = torchvision.models.convnext_tiny(weights="DEFAULT")   
        modules=list(full_cnn.children())[:-2]
        self.feature_extractor=nn.Sequential(*modules)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.dim_features_feature_extractor = 768
        self.n_features_feature_extractor = 49 # 7x7
        self.dim_text_features = 300
        # Dimension in which the images and text are embedded
        self.dim = 350

        # Embed for the text and image features
        self.cnn_features_embed = nn.Linear(self.n_features_feature_extractor, self.dim)
        self.text_features_embed = nn.Linear(self.dim_text_features, self.dim)

        # Positional embedding for the image features
        self.pos_embedding = nn.Parameter(torch.randn(1, self.dim_features_feature_extractor + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=heads_transformer, dim_feedforward=dim_fc_transformer, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth_transformer)

        # Classification fc
        self.fc = nn.Sequential(
            nn.Linear(self.dim, dim_fc_transformer),
            nn.GELU(),
            nn.Linear(dim_fc_transformer, num_classes)
        )

    def forward(self, img, txt):
        batch_size = img.shape[0]

        image_features = self.feature_extractor(img)
        image_features = image_features.reshape(batch_size, self.n_features_feature_extractor, self.dim_features_feature_extractor).permute(0, 2, 1)
        image_features = self.cnn_features_embed(image_features) 

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, image_features), dim=1)
        x += self.pos_embedding

        text_features = self.text_features_embed(txt.float())
        x = torch.cat((x, text_features), dim=1)
        x = self.transformer(x)

        x = x[:, 0]
        x = self.fc(x)
        return x