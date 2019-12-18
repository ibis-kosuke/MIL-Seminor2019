import torch
import torch.nn as nn

import os
import time

class Trainer():
    def __init__(self, opt, train_dataset, val_dataset, writer=None):
        self.out_dir = opt.out_dir
        self.expt_dir = opt.expt_dir
        self.epoch = opt.epoch
        self.print_every = 100
        self.train_iter_num = round(len(train_dataset)/ opt.batch_size)
        self.val_iter_num = round(len(val_dataset)/ opt.batch_size)
        self.train_losses = []
        self.val_losses = []
        self.writer=writer
        
    def train_epoch(self, epoch, dataloader, model, optimizer, criterion):
        print('Training step')
        model.train()
        data_iter = iter(dataloader)
        count = 0
        all_count = 0
        running_loss = 0.0
        epoch_loss = 0.0
        for j in range(self.train_iter_num):
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
                print('train_iteration: {}/{}'.format(j+1, self.train_iter_num))
                print('train_loss: {}'.format(running_loss / count))
                self.train_losses.append(running_loss / count)
                all_count += count
                running_loss = 0.0
                count = 0

            count += 1

        all_count += count

        #if self.writer is not None:
            #self.writer.add_scalar('Train_loss', epoch_loss/all_count, epoch+1)

        return  epoch_loss/all_count


    def val_epoch(self, epoch, dataloader, model, criterion, train_loss):
        print('Validation step')
        model.eval()
        data_iter = iter(dataloader)
        all_loss = 0.0
        all_acc = 0
        count = 0
        for i in range(self.val_iter_num):
            fnames, labels, imgs = data_iter.next()
            if torch.cuda.is_available():
                    labels = labels.cuda()
                    imgs = imgs.cuda()

            preds = model(imgs)
            loss = criterion(preds, labels)
            all_loss += loss.item()
            sample_num = preds.size(0)

            _, pred_labels = preds.max(dim=1)
            all_acc += int((pred_labels==labels).sum())
            count += sample_num

        self.val_losses.append(all_loss)
        print('val_accuracy: {}'.format(all_acc / count))
        print('val_loss:{}\n'.format(all_loss / count))

        if self.writer is not None:
            self.writer.add_scalars('Train_Val_loss',
                                    {'train_loss' : train_loss,
                                     'val_loss': all_loss / count} , epoch+1)

            #self.writer.add_scalar('Val_loss', all_loss/ count, epoch+1)
            self.writer.add_scalar('Val_acc', all_acc/ count, epoch+1)


    def train(self, train_loader, val_loader, model, optimizer, criterion):
        for epoch in range(self.epoch):
            print('Epoch: %d' % (epoch+1))
            train_loss=self.train_epoch(epoch, train_loader, model, optimizer, criterion)
            self.val_epoch(epoch, val_loader, model, criterion, train_loss)


        save_dir = os.path.join(self.out_dir, self.expt_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('Finished Training')
        print('saving model to {}'.format(save_dir))
        
        model_path = os.path.join(save_dir,'VGGnet.pth')
        torch.save(model.state_dict(), model_path)

        return self.train_losses, self.val_losses

