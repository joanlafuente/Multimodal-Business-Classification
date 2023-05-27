import torch.nn as nn
import torchvision
import torch
import math
# from einops import rearrange
# Conventional and convolutional neural network

class Transformer(nn.Module):
    def __init__(self, num_classes, depth_transformer, heads_transformer, dim_fc_transformer, drop=0.1):
        super(Transformer, self).__init__()

        full_cnn = torchvision.models.convnext_tiny(weights="DEFAULT")
        # full_cnn = torchvision.models.mobilenet_v3_large(weights="DEFAULT")
        # weights = torchvision.models.RegNet_Y_16GF_Weights.IMAGENET1K_V2
        # full_cnn  = torchvision.models.regnet_y_16gf(weights=weights) # output 3024
        # full_cnn = torchvision.models.efficientnet_b0(weights="DEFAULT")
        modules=list(full_cnn.children())[:-2]
        self.feature_extractor=nn.Sequential(*modules)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.dim_features_feature_extractor = 768 # convext_tiny
        # self.dim_features_feature_extractor = 960 # mobilenet
        self.n_features_feature_extractor = 49 # 7x7
        self.dim_text_features = 300
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Dimension in which the images and text are embedded
        self.dim = 360

        # Embed for the text and image features
        self.cnn_features_embed = nn.Linear(self.n_features_feature_extractor, self.dim)
        self.text_features_embed = nn.Linear(self.dim_text_features, self.dim)

        # Positional embedding for the image features
        self.pos_embedding = nn.Parameter(torch.randn(1, self.dim_features_feature_extractor + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=heads_transformer, dim_feedforward=dim_fc_transformer, batch_first=True,  dropout=drop)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth_transformer)

        # Classification fc
        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(self.dim, dim_fc_transformer),
            nn.Dropout(drop),
            nn.GELU(),
            nn.Linear(dim_fc_transformer, num_classes)
        )



    def forward(self, img, txt, text_mask):
        batch_size = img.shape[0]

        image_features = self.feature_extractor(img)
        image_features = image_features.reshape(batch_size, self.n_features_feature_extractor, self.dim_features_feature_extractor).permute(0, 2, 1)
        image_features = self.cnn_features_embed(image_features) 

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, image_features), dim=1)
        x += self.pos_embedding

        text_features = self.text_features_embed(txt.float())
        x = torch.cat((x, text_features), dim=1)

        tmp_mask = torch.zeros((img.shape[0], 1+self.dim_features_feature_extractor), dtype=torch.bool).to(self.device)
        mask = torch.cat((tmp_mask, text_mask), dim=1)
        x = self.transformer(x, src_key_padding_mask=mask)
        # x = self.transformer(x)

        x = x[:, 0]
        x = self.fc(x)
        return x
    



class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len, device):
        super().__init__()
        self.d_model = d_model
        self.device = device
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + nn.Parameter(self.pe[:,:seq_len], requires_grad=False).to(self.device)
        return x

class Transformer_positional_encoding_not_learned(nn.Module):
    def __init__(self, num_classes, depth_transformer, heads_transformer, dim_fc_transformer, drop=0.1):
        super(Transformer_positional_encoding_not_learned, self).__init__()

        full_cnn = torchvision.models.convnext_tiny(weights="DEFAULT")
        modules=list(full_cnn.children())[:-2]
        self.feature_extractor=nn.Sequential(*modules)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.dim_features_feature_extractor = 768 # convext_tiny
        # self.dim_features_feature_extractor = 960 # mobilenet
        self.n_features_feature_extractor = 49 # 7x7
        self.dim_text_features = 300
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Dimension in which the images and text are embedded
        self.dim = 360

        # Embed for the text and image features
        self.cnn_features_embed = nn.Linear(self.n_features_feature_extractor, self.dim)
        self.text_features_embed = nn.Linear(self.dim_text_features, self.dim)

        # Positional embedding for the image features
        self.pos_embedding = PositionalEncoder(self.dim, self.dim_features_feature_extractor + 1, self.device)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=heads_transformer, dim_feedforward=dim_fc_transformer, batch_first=True, dropout=drop)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth_transformer)

        # Classification fc
        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(self.dim, dim_fc_transformer),
            nn.Dropout(drop),
            nn.GELU(),
            nn.Linear(dim_fc_transformer, num_classes)
        )

    def forward(self, img, txt, text_mask):
        batch_size = img.shape[0]

        image_features = self.feature_extractor(img)
        image_features = image_features.reshape(batch_size, self.n_features_feature_extractor, self.dim_features_feature_extractor).permute(0, 2, 1)
        image_features = self.cnn_features_embed(image_features) 

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, image_features), dim=1)
        x = self.pos_embedding(x)

        text_features = self.text_features_embed(txt.float())
        x = torch.cat((x, text_features), dim=1)

        tmp_mask = torch.zeros((img.shape[0], 1+self.dim_features_feature_extractor), dtype=torch.bool).to(self.device)
        mask = torch.cat((tmp_mask, text_mask), dim=1)
        x = self.transformer(x, src_key_padding_mask=mask)
        # x = self.transformer(x)

        x = x[:, 0]
        x = self.fc(x)
        return x
    


class Feature_Extractor(nn.Module):
    def __init__(self):
        super(Feature_Extractor, self).__init__()
        full_cnn = torchvision.models.convnext_tiny(weights="DEFAULT")
        modules=list(full_cnn.children())[:-2]
        self.feature_extractor=nn.Sequential(*modules)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img):
        x = self.feature_extractor(img)
        return x


class Transformer_without_extracting_features(nn.Module):
    def __init__(self, num_classes, depth_transformer, heads_transformer, dim_fc_transformer):
        super(Transformer_without_extracting_features, self).__init__()
        self.dim_features_feature_extractor = 768 
        self.n_features_feature_extractor = 49 # 7x7
        self.dim_text_features = 300
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Dimension in which the images and text are embedded
        self.dim = 360

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

    def forward(self, image_features, txt, text_mask):
        batch_size = txt.size(0)
        image_features = image_features.reshape(batch_size, self.n_features_feature_extractor, self.dim_features_feature_extractor).permute(0, 2, 1)
        image_features = self.cnn_features_embed(image_features) 

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, image_features), dim=1)
        x += self.pos_embedding

        text_features = self.text_features_embed(txt.float())
        x = torch.cat((x, text_features), dim=1)

        tmp_mask = torch.zeros((batch_size, 1+self.dim_features_feature_extractor), dtype=torch.bool).to(self.device)
        mask = torch.cat((tmp_mask, text_mask), dim=1)
        x = self.transformer(x, src_key_padding_mask=mask)

        x = x[:, 0]
        x = self.fc(x)
        return x