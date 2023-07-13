from tqdm import tqdm
import tokenizers
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from torchmetrics import Accuracy, Recall

# logger = logging.getLogger(__name__)
from nltk.translate.bleu_score import corpus_bleu
from transformers import T5Tokenizer, T5ForConditionalGeneration
import csv
from transformers import ViltProcessor, ViltModel

class TrainerConfig:
    # optimization parameters
    max_epochs = 20
    batch_size = 24
    learning_rate = 1e-5
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False

    # checkpoint settings
    ckpt_path = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, val_dataset, test_dataset, optimizer=None, scheduler=None, config=None, args=None):
        # print(1)
        self.model = model
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=config.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=False)
        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)

    def save_checkpoint(self):
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            # logger.info("saving %s", self.config.ckpt_path)
            print("save_checkpoint")
            torch.save({
                'model_state_dict': ckpt_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
                }, self.config.ckpt_path)

    def save(self, writing_params_path):
        if writing_params_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            # logger.info("saving %s", writing_params_path)
            torch.save(ckpt_model.state_dict(), writing_params_path)
    
    def train(self, image_only=False, text_only=False):
        model, optimizer, scheduler, config = self.model, self.optimizer, self.scheduler, self.config
        # self.save_checkpoint()
        best_accur = 0.7475
        model.train()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, verbose=True)
        for epoch in range(config.max_epochs):
            total_loss = 0
            model.train()
            for batch in tqdm(self.train_dataloader):
                # print(inputs, label)
                image, text, label = batch

                # print(batch)
                optimizer.zero_grad()
                # print(image, list(text), label)
                outputs, loss = model([Image.open(image_path) for image_path in image], list(text), torch.tensor(label).to(self.device))

                loss.backward()
                optimizer.step()
                # print(loss)
                total_loss += loss.item()
            # 打印损失
            print(f"Epoch {epoch+1}/{config.max_epochs}, Loss: {total_loss/len(self.train_dataloader):.4f}")
            accur = self.eval(self.val_dataloader)
            # if epoch % 3 == 0:
            if accur > best_accur:
                best_accur = accur
                self.save_checkpoint()
            # print(f"Epoch {epoch+1}/{config.max_epochs}, Loss: {total_loss/len(self.train_dataloader):.4f}")

        # scheduler.step()
            
    def eval(self, dataloader, image_only=False, text_only=False, is_val=True):  # test -> is_val=False
        # 评估模型
        model, config = self.model, self.config
        model.eval()
        references = []
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                image, text, label = batch

                if is_val:
                    references.extend(label)

                outputs, _ = model([Image.open(image_path) for image_path in image], text, torch.tensor(label).to(self.device))
                
                predictions.extend(outputs)
                
        # 计算BLEU-4评估指标
        if is_val:
            acc = Accuracy(task="multiclass", num_classes=3)
            accur = acc(torch.tensor(references), torch.tensor(predictions))

            print(f"accur Score: {accur:.4f}")
            return accur
        else:
            # print(1)
            txt_file = 'tests_without_label.txt'
            replace_test_file(txt_file, predictions)

            
def replace_test_file(txt_file, predictions):
    # 读取CSV文件并将数据存储在列表中
    # print(1)
    predictions = ['positive' if value == 0 else 'neutral' if value == 1 else 'negative' for value in predictions]

    with open(txt_file, 'r') as file:
        lines = file.readlines()[1:]

    # Perform the replacements
    for i, line in enumerate(lines):
        lines[i] = line.replace('null', predictions[i])

    # Save the updated content back to the file
    with open('output.txt', 'w') as file:
        file.writelines(lines)