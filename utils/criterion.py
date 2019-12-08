import torch
import torch.nn.functional as F
import numpy as np

def accuracy_v1(preds, labels, top=[1,5]):
    """Compute the precision@k for the specified values of k"""
    correct = [0] * len(top)
    _, labels_pred = torch.sort(preds, dim=1, descending=True)
    for idx, label_pred in labels_pred:
        result = (label_pred == labels[idx])
        j = 0
        for i in range(top[-1]):
            while i-1 > top[j]:
                j += 1
            if result[i] == 1:
                while j < len(top):
                    correct[j] += (100.0 / len(preds))
                # end the loop
                break
    return correct


def accuracy_v2(preds, labels, top=[1,5]):
    """Compute the precision@k for the specified values of k"""

    maxk = max(top)
    batch_size = preds.size(0)

    _, pred = preds.topk(maxk, 1, True, True)
    pred = pred.t() # pred[k-1] stores the k-th predicted label for all samples in the batch.
    correct = pred.eq(labels.view(1,-1).expand_as(pred))

    k = 1
    correct_k = correct[:k].view(-1).float().sum(0)
    result = (correct_k.mul_(100.0 / batch_size))

    return result

def our_opt_loss(preds, soft_labels, use_cuda, args):
    
    # introduce prior prob distribution p
    if use_cuda:
        p = torch.ones(2).cuda() / 2
    
    #print(preds.size())
    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)

   
    L_o=torch.mean((F.softmax(preds)-soft_labels)*(F.softmax(preds)-soft_labels)/2)
    
    #L_o=torch.mean(torch.cosh(F.softmax(preds)-soft_labels))
    L_p = torch.sum(torch.log(prob_avg)*soft_labels )

    #L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))

    #L_c = -torch.mean(torch.sum(soft_labels * F.log_softmax(preds, dim=1), dim=1))

    loss = L_o 

    return prob, loss


