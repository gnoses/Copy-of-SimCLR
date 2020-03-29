import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model

import pickle
# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        
        features = []
        outputs = []
        targets = []
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)
            features.append(feature.cpu().data.numpy())
            outputs.append(out.cpu().data.numpy())
            targets.append(target.cpu().data.numpy())
            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(total_top1 / total_num * 100, total_top5 / total_num * 100))
        
        with open('output.bin','wb') as fp:
            pickle.dump((features, outputs, targets),fp)
            print('output dumped...')

    
    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    
    feature_dim = 128
    batch_size = 16
    k = 200
    temperature = 0.5
    c = 10
    memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)    
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False)
    
    test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = Model(feature_dim).cuda()
    device = torch.device("cuda")
    model_path = 'results/model.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    
    test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        
