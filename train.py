import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
import argparse
import logging
import os
import time
from dataset.medical_2 import MEDICAL
#from utils.visualize import save_fig
from utils.criterion import accuracy_v2, our_opt_loss
from utils.AverageMeter import AverageMeter
import torch.utils.data as data
from torch import optim
import random
import torchvision
from models.alexnet import alexnet
from torch.optim.lr_scheduler import *
import torch.nn as nn
from torch.autograd import Variable
from utils.utilsinfo import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1,help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='#images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=20, help='training epoches')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--epoch_begin', default=2, help='the epoch to begin update labels')
    parser.add_argument('--epoch_update', default=2, help='#epoch to average to update soft labels')
    parser.add_argument('--gpus', type=str, default='0', help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', type=str, default='./data/model_data', help='Directory of the output')
    parser.add_argument('--download', type=bool, default=True, help='download dataset')
    parser.add_argument('--network', type=str, default='alexnet', help='the backbone of the network')
    args = parser.parse_args()
    return args

def data_config(args):
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]

    transform_train=transforms.Compose([
        torchvision.transforms.Resize(227),
        transforms.RandomCrop(224), 
        transforms.RandomVerticalFlip(p=0.2), 
        transforms.ToTensor(),

    ])

    transform_val=transforms.Compose([ 
        torchvision.transforms.Resize(227),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),

    ])

    trainset = MEDICAL(root='./data/medical/', argsInfo=args,       
                                    train=True, 
                                    transform=transform_train)

    valset = MEDICAL(root='./data/medical/', argsInfo=args, 
                                train=False, 
                                transform=transform_val)

    trainloader=torch.utils.data.DataLoader(trainset,batch_size=16,shuffle=True,num_workers=2)
    valloader=torch.utils.data.DataLoader(valset,batch_size=16,shuffle=False,num_workers=2)

    return trainloader, valloader


criterion=nn.CrossEntropyLoss()
criterion.cuda()

def network_config(args):
    model=alexnet(pretrained=True)
    #model=vgg16(pretrained=True)
    model.cuda()

    classifier_h_params = list(map(id, model.classifier_h.parameters())) 
    classifier_s_params = list(map(id, model.classifier_s.parameters()))

    ignored_params = classifier_h_params + classifier_s_params

    base_params = filter(lambda p: id(p) not in ignored_params,model.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.0001},
        {'params': model.classifier_h.parameters(), 'lr': 0.01},
        {'params': model.classifier_s.parameters(), 'lr': 0.01}], 0.01, momentum=0.9, weight_decay=1e-3)

    scheduler=StepLR(optimizer,step_size=4, gamma=0.1)
   
    return model, optimizer, scheduler, True


def save_checkpoint(state, epoch):
    dst = 'models/checkpoint/epoch-' + str(epoch) + '.pkl'
    torch.save(state, dst)


def train(train_loader, network, optimizer, scheduler, use_cuda, args):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    scheduler.step()
    network.train()

    end = time.time()

    results = np.zeros((len(train_loader.dataset), 2), dtype=np.float32)
    for batch_idx,(images, labels_h, labels_s, soft_labels, index) in enumerate(train_loader):
 
        images = Variable(images.cuda())
        labels_h = Variable(labels_h.cuda())
        soft_labels = Variable(soft_labels.cuda())
        index = Variable(index.cuda())
        labels_s = Variable(labels_s.cuda())
        optimizer.zero_grad()
        

        output_h, output_s = network(images)
        prob, loss_s_soft = our_opt_loss(output_s, soft_labels, use_cuda, args)

        results[index.cpu().detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        prec1 = accuracy_v2(output_h, labels_h, top=[1,2])
        train_loss.update(loss_s_soft.item(), images.size(0))
        top1.update(prec1, images.size(0))
  
        loss_h=criterion(output_h, labels_h)
        loss_s=criterion(torch.nn.functional.log_softmax(output_s,dim=1), labels_h)


        loss= 0.35*loss_h + 5.0*loss_s_soft + 0.7*loss_s 


        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # update soft labels
    train_loader.dataset.update_labels(results)
    return train_loss.avg, top1.avg,  batch_time.sum


def validate(val_loader, network, criterion, use_cuda):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    top1 = AverageMeter()
    val_loss = AverageMeter()

    # switch to evaluate mode
    network.eval()

    total=0
    correct=0
    total_malignant1 = 0
    total_benign1 = 0
    pre_malignant1 = 0
    pre_benign1 = 0
    correct_malignant1 = 0
    correct_benign1 = 0
    statistics_dict={}
    for i in range(0, statistic_type.statistic_type_max):
        statistics_dict[i] = 0



    with torch.no_grad():
        end = time.time()
        for batch_idx,(images, labels_h, labels_s, soft_labels, index) in enumerate(val_loader):
            if use_cuda:
                images = images.cuda()
                labels_h = labels_h.cuda(non_blocking=True)
            outputs_h, outputs_s = network(images)
            prec1 = accuracy_v2(outputs_h, labels_h, top=[1,1])
            loss = criterion(outputs_h, labels_h)

            top1.update(prec1, images.size(0))
            val_loss.update(loss.item(), images.size(0))

            _,predicted=torch.max(outputs_h.data,1)
            total+=images.size(0)
            correct+=predicted.data.eq(labels_h.data).cpu().sum()
            
            statistics_result(predicted, labels_h, statistics_dict)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(" Acc: %f"% ((1.0*correct.numpy())/total))
    compute_result("net result", statistics_dict)

    return top1.avg, batch_time.sum


def main(args):
    # best_ac only record the best top1_ac for validation set.
    best_ac = 0.0

    # data lodaer
    train_loader, val_loader = data_config(args)

    # criterion
    val_criterion = nn.CrossEntropyLoss()

    # network config
    network, optimizer, scheduler, use_cuda = network_config(args)

    for epoch in range(args.epoch):
        # train for one epoch
        train_loss, top1_train_ac, train_time = train(train_loader, network, optimizer, scheduler, use_cuda, args)
        # evaluate on validation set
        top1_val_ac, val_time = validate(val_loader, network, val_criterion, use_cuda)
        # remember best prec@1, save checkpoint and logging to the console.
        if top1_val_ac >= best_ac:
            state = {'state_dict': network.state_dict(), 'epoch': epoch, 'ac': [top1_val_ac], 'best_ac': best_ac, 'time': [train_time, val_time]}
            best_ac = top1_val_ac
            # save model
            save_checkpoint(state, epoch)
            # logging
        logging.info('Epoch: [{}|{}], train_loss: {:.3f}, top1_train_ac: {:.3f}, top1_val_ac: {:.3f}, val_time: {:.3f}, train_time: {:.3f}'.format(epoch, args.epoch, train_loss, top1_train_ac, top1_val_ac, val_time, train_time))

    #save_fig(dst_folder)
    print('Best ac:%f'%best_ac)


if __name__ == "__main__":
    args = parse_args()
    logging.info(args)

    # train
    main(args)
