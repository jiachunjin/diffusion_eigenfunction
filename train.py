import os
import shutil
import logging
import argparse
import torch
from tensorboard import program

from trainer.score_trainer import Score_Trainer
from trainer.neuralef_trainer import NeuralEF_Trainer
from utils.experiment import create_exp_name, load_config, str2bool


def main(args):
    config, log_dir, ckpt_dir, sample_dir = create_experiment(args)
    my_logger = get_logger(level = getattr(logging, args.verbose.upper(), None))
    if str2bool(args.tensorboard):
        setup_tensorboard(args, log_dir)
    # ================================================================================
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    my_logger.info(f'Using device: {device}')
    config['device'] = device
    # ================================================================================
    config['data']['name'] = args.data
    if args.mode == 'score':
        trainer = Score_Trainer(config, my_logger, log_dir, ckpt_dir, sample_dir)
    elif args.mode == 'nef':
        trainer = NeuralEF_Trainer(config, my_logger, log_dir, ckpt_dir, sample_dir)

    my_logger.info('Start to train')
    trainer.train()
    my_logger.info('Training Done')

def get_logger(level):
    handler1 = logging.StreamHandler()
    # handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    # handler2.setFormatter(formatter)
    my_logger = logging.getLogger('training_logger')
    my_logger.addHandler(handler1)
    # logger.addHandler(handler2)
    my_logger.setLevel(level)
    return my_logger

def create_experiment(args):
    config_path = os.path.join('config', f'{args.config}.yaml')
    config = load_config(config_path)
    log_dir = ckpt_dir = sample_dir = None
    name = create_exp_name(args.exp_name)
    exp_dir = os.path.join('experiments', name)
    log_dir = os.path.join('./', exp_dir, 'logs')
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    sample_dir = os.path.join(exp_dir, 'samples')
    os.makedirs(exp_dir)
    os.makedirs(log_dir)
    os.makedirs(ckpt_dir)
    os.makedirs(sample_dir)
    shutil.copyfile(config_path, os.path.join(exp_dir, f'{args.config}.yaml'))
    return config, log_dir, ckpt_dir, sample_dir

def setup_tensorboard(args, log_dir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, f'--logdir={log_dir}', f'--port={args.tb_port}', f'--load_fast=false'])
    url = tb.launch()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str, required=True)
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['score', 'nef'])
    parser.add_argument('-d', '--data', type=str, choices=['toy_sin_a', 'toy_sin_b', 'toy_sin_pair'], required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--tensorboard', type=str, default='True')
    parser.add_argument('--tb_port', type=int, default=9990)
    parser.add_argument('--verbose', type=str, default='info')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)

"""
python train.py -m score -n score_sin_a -d toy_sin_a -c score_toy_2d --verbose info
python train.py -m score -n score_sin_b -d toy_sin_b -c score_toy_2d --verbose info --tb_port 9991

python train.py -m nef -n tem -d toy_sin_pair -c nef_toy_2d --verbose info
"""