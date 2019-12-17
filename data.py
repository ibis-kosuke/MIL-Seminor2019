import torch
import os
import numpy as np
import pickle as pkl
from collections import defaultdict


class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, split='train', data_dir='/data/unagi0/cifar/cifar-100-python'):
        super(CifarDataset, self).__init__()
        print('split: %s' % split)
        data_path = os.path.join(data_dir, split)
        with open(data_path, 'rb') as f:
            train_dic = pkl.load(f, encoding='latin1')
        
        data_dic=defaultdict(list)

        self.filenames = train_dic['filenames']
        self.fine_labels = train_dic['fine_labels']
        datas = []
        for data in train_dic['data']:
            data = data.reshape(3, 32, 32)
            data = data.swapaxes(0,2)
            data = data.swapaxes(0,1)
            datas.append(transform(data))
        self.datas = datas

        '''
        print(len(self.filenames))
        print(len(self.fine_labels))
        print(len(self.datas))
        '''

        '''
        for fname, label, array in zip(train_dic['filenames'], 
                        train_dic['fine_labels'], train_dic['data']):
            #ch, x, y
            array = array.reshape(3, 32, 32)
            array = array.swapaxes(1,2)
            #Tenfor, ch, y, x
            array = array.from_numpy() ##
            data_dic[label].append((fname, array))

        print('len_num_label:{}'.format(len(data_dic)))
        #label: (fname, array)
        self.data_dic = data_dic ###array: 0~255?
        self.filenames = train_dic['filenames']
        '''
    
        
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        label = self.fine_labels[idx]
        #print('label_len_before:{}'.format(len(label)))
        #print('label_before:{}'.format(label))
        #label = torch.Tensor(label)
        '''
        print('label_size_after:{}'.format(label.size()))
        print('label_after:{}'.format(label))
        '''
        #img: Tensor 3 x 32 x 32
        img = self.datas[idx]

        '''
        print('label:{}'.format(label))
        print('label_size:{}'.format(label.size()))
        print('img:{}'.format(img))
        print('img_size:{}'.format(img.size()))
        '''

        return fname, label, img
    
    def __len__(self):
        return len(self.filenames)


    