import torch.nn as nn
import torchvision
import torch
import math


# The idea of how to implement the model in order to take into account the text and the images was taken from:
#https://github.com/lluisgomez/ConTextTransformer/blob/main/ConTextTransformer_inference.ipynb



class Transformer(nn.Module):
    def __init__(self, num_classes, depth_transformer, heads_transformer, dim_fc_transformer, drop=0.1):
        super(Transformer, self).__init__()

        # Nedded for the mask of the imgs, so it is in the same device as the rest of the tensors
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        full_cnn = torchvision.models.convnext_tiny(weights="DEFAULT")

        # OTHER OPTIONS TRIED FOR THE VISUAL FEATURES EXTRACTION
        # full_cnn = torchvision.models.mobilenet_v3_large(weights="DEFAULT")
        # weights = torchvision.models.RegNet_Y_16GF_Weights.IMAGENET1K_V2
        # full_cnn  = torchvision.models.regnet_y_16gf(weights=weights) # output 3024
        # full_cnn = torchvision.models.efficientnet_b0(weights="DEFAULT")
        
        # Removing the average pooling and fully conected of the model
        modules=list(full_cnn.children())[:-2]
        self.feature_extractor=nn.Sequential(*modules)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True # Setting the CNN to be trainable, in this way we are finetuning it for the specific task
        
        # Dimension of the features extracted by the CNN
        self.dim_features_feature_extractor = 768 # number of feature maps
        self.n_features_feature_extractor = 49 # (7x7) Size of the feature maps flattened
      
        self.dim_text_features = 300 # Dimension of the text features
        self.dim = 360 # Dimension in which the images and text features are embedded

        # Linear layer to embed the text and images into the same space
        self.cnn_features_embed = nn.Linear(self.n_features_feature_extractor, self.dim)
        self.text_features_embed = nn.Linear(self.dim_text_features, self.dim)

        # Positional embedding for the image features and CLS token, its a learnable parameter so the model learns it during training
        self.pos_embedding = nn.Parameter(torch.randn(1, self.dim_features_feature_extractor + 1, self.dim))

        # CLS token, its a learnable parameter. So the model learns it during training
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=heads_transformer, dim_feedforward=dim_fc_transformer, batch_first=True,  dropout=drop)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth_transformer)

        # Classification MLP, that will get the output of the transformer in the CLS token to classify it
        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(self.dim, dim_fc_transformer),
            nn.Dropout(drop),
            nn.GELU(),
            nn.Linear(dim_fc_transformer, num_classes)
        )


    def forward(self, img, txt, text_mask):
        batch_size = img.shape[0] # We get the batch size passed to the model

        # Extract the features from the images
        image_features = self.feature_extractor(img) # Shape (batch_size, 768, 7, 7)
        # Flatten the feature maps and permute the dimensions to get the right shape for the embedding
        image_features = image_features.reshape(batch_size, self.n_features_feature_extractor, self.dim_features_feature_extractor).permute(0, 2, 1) # Shape (batch_size, 49, 768)
        # Projecting the feature maps into the same space as the text features
        image_features = self.cnn_features_embed(image_features) # Shape (batch_size, self.dim, 768)

        # We add the CLS token for the transformer at the first position
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Shape (batch_size, self.dim, 1)
        x = torch.cat((cls_tokens, image_features), dim=1) # Shape (batch_size, self.dim, 769)
        # We add the positional embedding to the image features and CLS token
        x += self.pos_embedding

        # Projecting the text features into the same space as the image features
        text_features = self.text_features_embed(txt.float())
        x = torch.cat((x, text_features), dim=1) # Shape (batch_size, self.dim, 769 + max_num_words)

        # Create a mask of zeors for the image features and CLS token, so all are taken into account by the transformer
        tmp_mask = torch.zeros((batch_size, 1+self.dim_features_feature_extractor), dtype=torch.bool).to(self.device)
        mask = torch.cat((tmp_mask, text_mask), dim=1) # We concatenate it with the mask of the word features, which is 1 for the words of the padding
        
        # Pass the features and mask through the transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Getting the output of the transformer encoder for the CLS token and passing it through a MLP to have the dimension
        # equal to the number of clases and be able to classify
        x = x[:, 0, :]
        x = self.fc(x)
        return x
    


# Code for the positional encoding taken from https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len, device):
        super().__init__()
        self.d_model = d_model
        self.device = device
        # create constant 'pe' matrix
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                # Get the positional encoding for each position of the sequence
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
        # Nedded for the mask of the imgs, so it is in the same device as the rest of the tensors
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        full_cnn = torchvision.models.convnext_tiny(weights="DEFAULT")
        # Removing the average pooling and fully conected of the model
        modules=list(full_cnn.children())[:-2]
        self.feature_extractor=nn.Sequential(*modules)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True # Setting the CNN to be trainable, in this way we are finetuning it for the specific task
        
        self.dim_features_feature_extractor = 768 # number of feature maps
        self.n_features_feature_extractor = 49 # (7x7) Size of the feature maps flattened
    
        self.dim_text_features = 300 # Dimension of the text features
        self.dim = 360 # Dimension in which the images and text features are embedded

        # Linear layer to embed the text and images into the same space
        self.cnn_features_embed = nn.Linear(self.n_features_feature_extractor, self.dim)
        self.text_features_embed = nn.Linear(self.dim_text_features, self.dim)

        # Positional embedding for the image features and CLS token, in this case is constant (Not learned)
        self.pos_embedding = PositionalEncoder(self.dim, self.dim_features_feature_extractor + 1, self.device)

        # CLS token, its a learnable parameter. So the model learns it during training
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=heads_transformer, dim_feedforward=dim_fc_transformer, batch_first=True, dropout=drop)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth_transformer)

        # Classification MLP, that will get the output of the transformer in the CLS token to classify it
        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(self.dim, dim_fc_transformer),
            nn.Dropout(drop),
            nn.GELU(),
            nn.Linear(dim_fc_transformer, num_classes)
        )

    def forward(self, img, txt, text_mask):
        batch_size = img.shape[0] # We get the batch size passed to the model

        # Extract the features from the images
        image_features = self.feature_extractor(img) # Shape (batch_size, 768, 7, 7)
        # Flatten the feature maps and permute the dimensions to get the right shape for the embedding
        image_features = image_features.reshape(batch_size, self.n_features_feature_extractor, self.dim_features_feature_extractor).permute(0, 2, 1) # Shape (batch_size, 49, 768)
        # Projecting the feature maps into the same space as the text features
        image_features = self.cnn_features_embed(image_features)  # Shape (batch_size, self.dim, 768)

        # We add the CLS token for the transformer at the first position
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Shape (batch_size, self.dim, 1)
        x = torch.cat((cls_tokens, image_features), dim=1) # Shape (batch_size, self.dim, 769)
        # We add the positional embedding to the image features and CLS token
        x = self.pos_embedding(x)

        # Projecting the text features into the same space as the image features
        text_features = self.text_features_embed(txt.float())
        x = torch.cat((x, text_features), dim=1) # Shape (batch_size, self.dim, 769 + max_num_words)

        # Create a mask of zeros for the image features and CLS token, so all are taken into account by the transformer
        tmp_mask = torch.zeros((batch_size, 1+self.dim_features_feature_extractor), dtype=torch.bool).to(self.device)
        mask = torch.cat((tmp_mask, text_mask), dim=1)

        # Pass the features and mask through the transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Getting the output of the transformer encoder for the CLS token and passing it through a MLP to have the dimension
        # equal to the number of clases and be able to classify
        x = x[:, 0, :]
        x = self.fc(x)
        return x
    

# A simple model that given an image gives the features extracted by a CNN, in this case convnext_tiny
# Its used as a feature extractor, so the parameters cannot be learned
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
        # Nedded for the mask of the imgs, so it is in the same device as the rest of the tensors
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

        # Dimension of the features extracted by the CNN
        self.dim_features_feature_extractor = 768 # number of feature maps
        self.n_features_feature_extractor = 49 # (7x7) Size of the feature maps flattened


        self.dim_text_features = 300 # Dimension of the text features
        self.dim = 360 # Dimension in which the images and text features are embedded

        # Linear layer to embed the text and images into the same space
        self.cnn_features_embed = nn.Linear(self.n_features_feature_extractor, self.dim)
        self.text_features_embed = nn.Linear(self.dim_text_features, self.dim)

         # Positional embedding for the image features and CLS token, its a learnable parameter so the model learns it during training
        self.pos_embedding = nn.Parameter(torch.randn(1, self.dim_features_feature_extractor + 1, self.dim))

        # CLS token, its a learnable parameter. So the model learns it during training
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim, nhead=heads_transformer, dim_feedforward=dim_fc_transformer, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth_transformer)

        # Classification MLP, that will get the output of the transformer in the CLS token to classify it
        self.fc = nn.Sequential(
            nn.Linear(self.dim, dim_fc_transformer),
            nn.GELU(),
            nn.Linear(dim_fc_transformer, num_classes)
        )

    def forward(self, image_features, txt, text_mask):
        batch_size = txt.size(0)
        # Flatten the feature maps and permute the dimensions to get the right shape for the embedding
        image_features = image_features.reshape(batch_size, self.n_features_feature_extractor, self.dim_features_feature_extractor).permute(0, 2, 1) # Shape (batch_size, 49, 768)
        # Projecting the feature maps into the same space as the text features
        image_features = self.cnn_features_embed(image_features) # Shape (batch_size, self.dim, 768)

        # We add the CLS token for the transformer at the first position
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Shape (batch_size, self.dim, 1)
        x = torch.cat((cls_tokens, image_features), dim=1) # Shape (batch_size, self.dim, 769)
        # We add the positional embedding to the image features and CLS token
        x += self.pos_embedding

        # Projecting the text features into the same space as the image features
        text_features = self.text_features_embed(txt.float())
        x = torch.cat((x, text_features), dim=1) # Shape (batch_size, self.dim, 769 + max_num_words)

        # Create a mask of zeors for the image features and CLS token, so all are taken into account by the transformer
        tmp_mask = torch.zeros((batch_size, 1+self.dim_features_feature_extractor), dtype=torch.bool).to(self.device)
        mask = torch.cat((tmp_mask, text_mask), dim=1) # We concatenate it with the mask of the word features, which is 1 for the words of the padding
        
        # Pass the features and mask through the transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Getting the output of the transformer encoder for the CLS token and passing it through a MLP to have the dimension
        # equal to the number of clases and be able to classify
        x = x[:, 0, :]
        x = self.fc(x)
        return x
    

# Trying to use ViT (visual transformer) instead of CNN for the image features, This model does not work well, accuracy arround 50%
class Transformer_positional_encoding_not_learned_ViT(nn.Module):
    def __init__(self, num_classes, depth_transformer, heads_transformer, dim_fc_transformer, drop=0.1):
        super(Transformer_positional_encoding_not_learned, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # full_cnn = torchvision.models.convnext_tiny(weights="DEFAULT")
        
        # modules=list(full_cnn.children())[:-2]
        # self.feature_extractor=nn.Sequential(*modules)
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = True
        
        full_ViT = ViT('B_16_imagenet1k', pretrained=True)
        
        modules=list(full_ViT.children())[:-2]
        self.feature_extractor=nn.Sequential(*modules)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        
        self.dim_features_feature_extractor = 0
        self.n_features_feature_extractor = 49 # 7x7
        self.dim_text_features = 300 # dim text embedding vectors
        
        
        self.dim = 360 # Dimension in which the images and text are embedded

        # Embed for the text features
        self.text_features_embed = nn.Linear(self.dim_text_features, self.dim)
        
        # self.cnn_features_embed = nn.Linear(self.n_features_feature_extractor, self.dim)
        # self.vit_features_embed = nn.Linear(self.dim_features_feature_extractor, self.dim)
        # self.ViT = ViT(image_size = 256, patch_size = 32, num_classes = 1000, dim = 1024, depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1)

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
        # image_features = self.cnn_features_embed(image_features) 

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
    