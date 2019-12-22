import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F ##

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import optuna
import random

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
    parser.add_argument('--data_dir', default='/data/unagi0/ktokitake/cifar100/data')
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--cuda_ids', type=str, default='-1') ###
    parser.add_argument('--manual_seed', type=str)
    #parser.add_argument('')

    args = parser.parse_args()

    return args

'''
def plot_losses(losses):
    steps = np.arange(len(losses))
    plt.plot(steps, losses, label='losses')
    plt.legend()
    plt.show()
'''


def set_vggcfg():
    vggcfg = [64,64,'M',128,128,'M',256,256,256,256,'M']

    return vggcfg

#######Train and val#######
def train_val(opt, trial, transform, model, optim):
    train_dataset = CifarDataset(transform, 'train', opt.data_dir)
    val_dataset = CifarDataset(transform, 'val', opt.data_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers =2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers =2)

    if torch.cuda.is_available():
        model.cuda(device=opt.device)
    if len(opt.cuda_list) > 1:
        model = nn.DataParallel(model, opt.cuda_list)
        model.to(opt.device)

    #set Tensorboard writer 
    writer_path = os.path.join(opt.out_dir,opt.expt_dir, opt.writer_dir)
    writer = SummaryWriter(writer_path)

    #set criterion, optimizer
    criterion = nn.CrossEntropyLoss()
    #optim = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9,0.999)) 

    #acrtivates --> cross_entropy 
    trainer = Trainer(opt, train_dataset, val_dataset, writer)
    train_losses, val_losses, val_accs = trainer.train(trial, train_loader, val_loader, model, optim, criterion)


    train_loss_path = os.path.join(opt.out_dir, opt.expt_dir, 'train_losses.pkl')
    val_loss_path = os.path.join(opt.out_dir, opt.expt_dir, 'val_losses.pkl')
    with open(train_loss_path, 'wb') as f, open(val_loss_path, 'wb') as g:
        pkl.dump(train_losses, f)
        pkl.dump(val_losses, g)

    #plot_losses(training_losses)

    return val_accs

#######Test######
def test(opt):####
    model_path = os.path.join(opt.out_dir, opt.expt_dir, 'VGGnet.pth')
    print('load model from {}'.format(model_path))

    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    if torch.cuda.is_available():
        model.cuda(device=device)       

    test_dataset = CifarDataset(transform, 'test', opt.data_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers =2)     
        
    evaluator = Evaluator(opt, test_dataset)
    evaluator.evaluate(test_loader, model)


def suggest_optimizer(trial, model):
    optim_str = trial.suggest_categorical('optimizer', ['Adam', 'MomentumSGD'])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    
    if optim_str == 'Adam':
        adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
        optim = torch.optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    
    else:
        sgd_lr = trial.suggest_loguniform('sgd_lr', 1e-5, 1e-1)
        optim = torch.optim.SGD(model.parameters(), lr=sgd_lr, momentum=0.9,
                                    weight_decay=weight_decay)

    return optim


def suggest_activation(trial):
    activation = trial.suggest_categorical('activation', ['ReLU', 'ELU'])

    if activation == 'ReLU':
        activation = nn.ReLU(True)
    else:
        activation = nn.ELU(True)

    return activation


def objective_wrapper(opt):

    def objective(trial):
        '''
        mean = trial.suggest_uniform('mean', 0, 1)
        var = trial.suggest_uniform('val', 1e-3, 1)
        '''

        r_mean = trial.suggest_uniform('r_mean', 1e-10, 1)
        g_mean = trial.suggest_uniform('g_mean', 1e-10, 1)
        b_mean = trial.suggest_uniform('b_mean', 1e-10, 1)
        r_var = trial.suggest_uniform('r_val', 1e-3, 1)
        g_var = trial.suggest_uniform('g_val', 1e-3, 1)
        b_var = trial.suggest_uniform('b_val', 1e-3, 1)

        #ToTensor: (hight x width x ch) to (ch x hight x width)  &  0~255 to 0~1
        #Normalize: 0~1 to -1~1
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((r_mean, g_mean, b_mean),(r_var, g_var, b_var))])
        '''
        activation = suggest_activation(trial)
        drop_prob = trial.suggest_uniform('drop_prob', 0, 0.8)
        '''
        activation = nn.ReLU(True)
        drop_prob = 0.27
    
        vggcfg = set_vggcfg()
        model = VGGnet(vggcfg, activation, drop_prob)

        #optim = suggest_optimizer(trial, model)
        #optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=1.9)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)


        #train and val
        val_accs = train_val(opt, trial, transform, model, optimizer)

        return 1.0 - val_accs[-1]

    return objective


if __name__ =='__main__':
    opt = parse_args()
    
    if opt.is_training:
        opt.shuffle = True
        opt.manual_seed = random.randint(0,10000)
        
    else:
        opt.shuffle = False
        opt.manual_seed = 100

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    opt.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.cuda_ids != '-1':
        torch.cuda.manual_seed_all(opt.manual_seed)
        opt.cuda_list = [int(gpu_id) for gpu_id in opt.cuda_ids.split(',')]
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_ids #make sense?
        #print('Available GPUs num:{}'.format(torch.cuda.device_count()))

        device = torch.device('cuda:%s'% str(opt.cuda_list[0]) if torch.cuda.is_available() else 'cpu')
        opt.device=device

    study = optuna.create_study()
    study.optimize(objective_wrapper(opt), n_trials=5)

    print('Finished trials')
    print('Best_trialnums:{}'.format(study.best_trial.number))
    print('Best_params;{}'.format(study.best_params))
    



    

    


    



    
