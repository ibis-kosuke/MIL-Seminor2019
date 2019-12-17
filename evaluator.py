import torch
import torch.nn as nn


class Evaluator():
    def __init__(self, opt, dataset):
        self.batch_size = opt.batch_size
        self.iter_num = round(len(dataset)/opt.batch_size)

    def evaluate(self, dataloader, model):
        model.eval()
        data_iter = iter(dataloader)
        all_acc = 0
        count = 0
        for i in range(self.iter_num):
            fnames, labels, imgs = data_iter.next()
            if torch.cuda.is_available():
                    labels = labels.cuda()
                    imgs = imgs.cuda()

            preds = model(imgs)
            sample_num = preds.size(0)

            _, pred_labels = preds.max(dim=1)
            #print('pred_labels_size:{}'.format(pred_labels.size()))
            all_acc += int((pred_labels==labels).sum())
            count += sample_num

        print('all_acc:{}, count:{}'.format(all_acc, count))
        print('test_accuracy: {}'.format(all_acc / count))
                
            

            



