import torch
import torch.nn as nn

import os
import time

class Trainer():
    def __init__(self, opt, dataset, writer=None):
        self.out_dir = opt.out_dir
        self.expt_dir = opt.expt_dir
        self.epoch = opt.epoch
        self.print_every = 100
        self.sample_num = len(dataset)
        self.iter_num = round(len(dataset)/ opt.batch_size)
        self.train_losses = []
        self.writer=writer
        
    
    def train(self, dataloader, model, optimizer, criterion):
        for epoch in range(self.epoch):
            print('Epoch: %d' % (epoch+1))
            data_iter = iter(dataloader)###position 
            count = 0
            running_loss = 0.0
            epoch_loss = 0.0
            for j in range(self.iter_num):
                optimizer.zero_grad()
                #labels: batch 
                #imgs: batch x Ch x hight x width
                fnames, labels, imgs = data_iter.next()
                if torch.cuda.is_available():
                    labels = labels.cuda()
                    imgs = imgs.cuda()
                
                preds = model(imgs)
                
                #criterion: nn.CrossEntropyLoss()
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
               
                running_loss += loss.item()
                epoch_loss += loss.item()

                if ((j+1) % self.print_every) == 0: 
                    print('iteration: {}/{}'.format(j+1, self.iter_num))
                    print('loss: {}'.format(running_loss / count))
                    self.train_losses.append(running_loss / count)
                    running_loss = 0.0
                    count = 0

                count += 1

            if self.writer is not None:
                self.writer.add_scalar('Train_loss', epoch_loss/self.sample_num, epoch+1)
                

        save_dir = os.path.join(self.out_dir, self.expt_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('Finished Training')
        print('saving model to {}'.format(save_dir))
        
        model_path = os.path.join(save_dir,'VGGnet.pth')
        torch.save(model.state_dict(), model_path)

        return self.train_losses

