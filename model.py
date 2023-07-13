import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoImageProcessor, ViTModel
from transformers import ViltProcessor, ViltModel, ViltConfig, AutoTokenizer, BertModel, BertConfig, ViTConfig
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
from torchvision import transforms

# 定义模型类
class Vilt(nn.Module):
    def __init__(self):
        super(Vilt, self).__init__()
        config = ViltConfig(max_position_embeddings=200)

        self.transformer = ViltModel.from_pretrained("dandelin/vilt-b32-mlm", config=config, ignore_mismatched_sizes=True)
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(768, 3)

        
    def forward(self, inputs, labels):
        # 使用transformer进行前向传播
 

        loss = None
        outputs = self.transformer(**inputs)
        outputs = self.dropout(outputs.pooler_output)
        logits = self.classifier(outputs)
        
        if labels[0] != -1:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        # print(outputs)
        # print(outputs.shape, labels)   
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=-1)
        preds = preds.tolist()
        # print(preds)
        return preds, loss
    
    
class BertVitMultiModel(nn.Module):
    def __init__(self):
        super(BertVitMultiModel, self).__init__()
        # config = ViltConfig(max_position_embeddings=200)
        self.BertTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.BertConfig = BertConfig.from_pretrained("bert-base-uncased")
        self.BertConfig.hidden_dropout_prob = 0.3  # Set dropout rate for BERT model
        self.BertModel = BertModel.from_pretrained("bert-base-uncased", config=self.BertConfig)

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize images to a larger size
            transforms.RandomCrop((224, 224)),  # Randomly crop images to the desired size
            transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
            transforms.ToTensor()  # Convert images to tensors
        ])

        vit_config = ViTConfig(dropout=0.3)
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", config=vit_config)
        self.VitModel = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", config=vit_config)
        
        # self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        # self.VitModel = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        self.dropout = nn.Dropout()
        self.classifier = nn.Sequential(
            nn.Linear(2*768, 768),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(768, 224),
            # nn.ReLU(),
            nn.Dropout(),
            nn.Linear(768, 3)
        )
        self.classifier_for_only = nn.Sequential(
            nn.Dropout(),
            nn.Linear(768, 3)
        )
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        
    def forward(self, image_inputs, text_inputs, labels):
        # 使用transformer进行前向传播
        logits = None
        if image_inputs == None:
          text_inputs = self.BertTokenizer(text_inputs, return_tensors="pt", padding='longest').to(self.device)
          text_outputs = self.BertModel(**text_inputs).pooler_output
          logits = self.classifier_for_only(text_outputs)

        elif text_inputs == None:
          image_inputs = [self.image_transform(image) for image in image_inputs]  # Apply the transform to each image        
          image_inputs = self.image_processor(image_inputs, return_tensors="pt").to(self.device)
          image_outputs = self.VitModel(**image_inputs).pooler_output

          
          # outputs = self.dropout(outputs)
          logits = self.classifier_for_only(image_outputs)
        else:
          text_inputs = self.BertTokenizer(text_inputs, return_tensors="pt", padding='longest').to(self.device)
          image_inputs = [self.image_transform(image) for image in image_inputs]  # Apply the transform to each image        
          image_inputs = self.image_processor(image_inputs, return_tensors="pt").to(self.device)
          
          text_outputs = self.BertModel(**text_inputs).pooler_output
          image_outputs = self.VitModel(**image_inputs).pooler_output


          # print(text_outputs.shape, image_outputs.shape)
          outputs = torch.concat((text_outputs, image_outputs), dim=1)
          
          # outputs = self.dropout(outputs)
          logits = self.classifier(outputs)
        
        
        loss = None 

        if labels[0] != -1:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        # print(outputs)
        # print(outputs.shape, labels)   
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=-1)
        preds = preds.tolist()
        # print(preds)
        return preds, loss