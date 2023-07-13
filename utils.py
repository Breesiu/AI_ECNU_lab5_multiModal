
import argparse


def build_args():
    
    args = argparse.ArgumentParser()
    
    #task
    args.add_argument('--do_eval',default=True)
    args.add_argument('--do_pretrain',default=False)
    args.add_argument('--do_finetune',default=True)
    args.add_argument('--do_inference',default=True)
    args.add_argument('--do_attack',default=False)
    
    args.add_argument('--ckpt_path_pretrain', default='./checkpoints/pretrain.pt')
    args.add_argument('--ckpt_path_fine_tune', default='./checkpoints/fine_tune.pt') #'./checkpoints/fine_tune.pt'

    args.add_argument('--writing_params_path_pretrain', default=None)
    args.add_argument('--writing_params_path_fine_tune', default='./results/fine_tune.pt')

    args = args.parse_args()
    print(1)
    return args
