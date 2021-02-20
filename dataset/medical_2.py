from __future__ import print_function
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs

#from .utils import noisify

#root="/home/caozhantao/Co-teaching/Co-teaching/"

# -----------------ready the dataset--------------------------
def default_loader(path):
    im = Image.open(path).convert('RGB')
    out = im.resize((227, 227))
    #out = im.resize((32, 32))
    #out = im.resize((180, 180))
    return out
    #.convert('RGB')


class_num = 2

class MEDICAL(data.Dataset):
    def __init__(self, root, argsInfo,train=True,
                 transform=None, target_transform=None,
                 loader=default_loader, aug = None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        #print (self.target_transform)
        self.train = train  
        # training set or test set
        self.dataset='medical'
        self.loader = loader
        self.aug = aug
        
        #self._num = int(len(self.train_labels) * args.noise_ratio)
        self._count = 1

        self.args = argsInfo

        if self.train:
            self.train_data, self.train_labels, self.bi_rads_labels = self.ReadImage(self.root, "train/", "train.txt")
            self.soft_labels = np.zeros((len(self.bi_rads_labels), class_num), dtype=np.float32)
            self.prediction = np.zeros((self.args.epoch_update, len(self.train_data), class_num) ,dtype=np.float32)

            #print(self.soft_labels)
            # to be more equal, every category can be processed separately
            idxes = np.random.permutation(len(self.bi_rads_labels))
            for i in range(len(idxes)):
                #print(self.bi_rads_labels[idxes[i]])
                l_value = 0
                e_value = 0
                if self.bi_rads_labels[idxes[i]] == 0:
                    l_value = 1.0
                    e_value = 0
                if self.bi_rads_labels[idxes[i]] == 1:
                    l_value = 0.985
                    e_value = 0.015

                if self.bi_rads_labels[idxes[i]] == 2:
                    l_value = 0.94
                    e_value = 0.06

                if self.bi_rads_labels[idxes[i]] == 3:
                    l_value = 0.355
                    e_value = 0.645

                if self.bi_rads_labels[idxes[i]] == 4:
                    l_value = 0.275
                    e_value = 0.725

                if self.bi_rads_labels[idxes[i]] == 5:
                    l_value = 0.025
                    e_value = 0.975

                self.soft_labels[idxes[i]][0] = l_value
                self.soft_labels[idxes[i]][1] = e_value

        else:
            self.test_data, self.test_labels, self.test_bi_labels = self.ReadImage(self.root, "val/", "val.txt")

        

    def ReadImage(self, root, middle, file_name):
        fh = open(root + middle + file_name, 'r')
        datas = []
        labels = []
        bi_rads_labels = []
        

        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            #print(words)
            image_path = root + middle + words[0]
            img = self.loader(image_path)
            datas.append(img)
            labels.append(int(words[1]))

            bi_label = 0
            if  "2" == words[2]:
                bi_label = 0
            if  "3" == words[2]:
                bi_label = 1
            elif "4a" == words[2]:
                bi_label = 2
            elif "4b" == words[2]:
                bi_label = 3
            elif "4c" == words[2]:
                bi_label = 4
            elif "5" == words[2]:
                bi_label = 5
               
            bi_rads_labels.append(bi_label)
          
            
        return datas, labels, bi_rads_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target, bi_rads_target, soft_labels = self.train_data[index], self.train_labels[index], self.bi_rads_labels[index],self.soft_labels[index]
        else:
            img, target, bi_rads_target, soft_labels = self.test_data[index], self.test_labels[index], self.test_bi_labels[index], self.test_bi_labels[index]

        if self.transform is not None:
            if self.aug is not None:
                
                img_array = np.asarray(img)
                #print ("source",img_array)
                student = self.aug.augment(img_array)
                #print ("target",student)
              
                img = Image.fromarray(np.uint8(student))

            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, bi_rads_target, soft_labels, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def update_labels(self, result):
        # use the average output prob of the network of the past [epoch_update] epochs as s.
        # update from [begin] epoch.

        idx = self._count % self.args.epoch_update
        self.prediction[idx,:] = result


        if self._count >= self.args.epoch_update:
            self.soft_labels = self.prediction.mean(axis = 0)

        self._count += 1

    def reload_labels(self):
        param = np.load(self.dst)
        self.train_data = param['data']
        self.train_labels = param['hard_labels']
        self.soft_labels = param['soft_labels']

