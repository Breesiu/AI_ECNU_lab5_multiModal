import torch
import pandas
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import corpus_bleu
from dataset import CustomDataset
from model import Vilt, BertVitMultiModel
from trainer import Trainer, TrainerConfig
from utils import build_args



if __name__ == '__main__':
    # TODO: random seed?
    train_label_path = 'train.txt'
    test_label_path = 'test_without_label.txt'
    data_dir = './data'
    
    args = build_args()
    
    dataset = CustomDataset(data_dir=data_dir, label_file=train_label_path)
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    test_dataset = CustomDataset(data_dir=data_dir, label_file=test_label_path)
    
    if args.do_pretrain:
        pass
    
    if args.do_finetune:
        
        tconfig = TrainerConfig()
        tconfig.ckpt_path = args.ckpt_path_fine_tune
        model = None
        model = BertVitMultiModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, verbose=True)
        # print(args.ckpt_path_fine_tune)
        if args.ckpt_path_fine_tune != None:
            # print("here1")
            #checkpoint = torch.load(args.ckpt_path_fine_tune)
            #model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            
            print("Checkpoint")
        elif args.writing_params_path_pretrain != None:
            checkpoint = torch.load(args.writing_params_path_pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # epoch = checkpoint['epoch']
        else:
            model = BertVitMultiModel()

        # print(train_dataset[:2])
        # print(test_dataset[:2])
        trainer = Trainer(model, train_dataset, val_dataset, test_dataset, optimizer=optimizer, scheduler=scheduler, config=tconfig, args=args)
        trainer.train()
        trainer.save(args.writing_params_path_fine_tune)
        
    if args.do_eval:
        model = BertVitMultiModel()
        tconfig = TrainerConfig()
        if args.ckpt_path_fine_tune != None:
            checkpoint = torch.load(args.ckpt_path_fine_tune)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        trainer = Trainer(model, train_dataset, val_dataset, test_dataset, config=tconfig)
        trainer.eval(trainer.val_dataloader)
        pass

    if args.do_inference:
        model = BertVitMultiModel()
        tconfig = TrainerConfig()
        if args.ckpt_path_fine_tune != None:
            checkpoint = torch.load(args.ckpt_path_fine_tune)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        trainer = Trainer(model, train_dataset, val_dataset, test_dataset, config=tconfig)
        trainer.eval(trainer.test_dataloader, is_val=False)    
        pass
    
    # print(1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    # print(1)
    # trainer = Trainer(model, train_dataset, val_dataset, TrainerConfig)
    # trainer.train()


