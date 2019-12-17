import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from trainer import Trainer 
from data import CifarDataset
from model import VGGnet
from evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--out_dir', default='/data/unagi0/ktokitake/cifar100')
    parser.add_argument('--expt_dir', default='expt')
    parser.add_argument('--writer_dir', default='tensorboard')
    parser.add_argument('--data_dir', default='/data/unagi0/cifar/cifar-100-python')
    parser.add_argument('--lr', type=float, default=0.002)
    #parser.add_argument('')

    args = parser.parse_args()

    return args

def plot_losses(losses):
    steps = np.arange(len(losses))
    plt.plot(steps, losses, label='losses')
    plt.legend()
    plt.show()


def set_vggcfg():
    vggcfg = [64,64,'M',128,128,'M',256,256,256,256,'M']

    return vggcfg


if __name__=='__main__':

    opt = parse_args()
    
    if opt.is_training:
        opt.split = 'train'
        opt.shuffle = True
    else:
        opt.split = 'test'
        opt.shuffle = False


    #######Training#######
    #ToTensor: (hight x width x ch) to (ch x hight x width)  &  0~255 to 0~1
    #Normalize: 0~1 to -1~1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    dataset = CifarDataset(transform, opt.split, opt.data_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers =2)

    vggcfg = set_vggcfg()
    model = VGGnet(vggcfg)
    
    if opt.is_training:
        model.train()
        if torch.cuda.is_available():
            model.cuda()

        #Tensorboard writer 
        writer_path = os.path.join(opt.out_dir,opt.expt_dir, opt.writer_dir)
        writer = SummaryWriter(writer_path)

        criterion = nn.CrossEntropyLoss()
        optim = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9,0.999)) 

        #acrtivates --> cross_entropy 
        trainer = Trainer(opt, dataset, writer)
        training_losses = trainer.train(dataloader, model, optim, criterion)

        pkl_path = os.path.join(opt.out_dir, opt.expt_dir, 'train_losses.pkl')
        with open(pkl_path, 'wb') as f:
            pkl.dump(training_losses, f)

        #plot_losses(training_losses)

    #######Test######
    else:
        model_path = os.path.join(opt.out_dir, opt.expt_dir, 'VGGnet.pth')
        print('load model from {}'.format(model_path))

        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            model.cuda()            
        model.eval()

        evaluator = Evaluator(opt, dataset)
        evaluator.evaluate(dataloader, model)
    
