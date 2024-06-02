import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops
from utils import add_self_loops
from tqdm import tqdm

from logger import Logger
from dataset import load_dataset
from data_utils import load_fixed_splits
from eval import evaluate, eval_acc, eval_rocauc, eval_f1
from parse import parse_method, parser_add_main_args

import warnings
warnings.filterwarnings('ignore')

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args)
if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

### get the splits for all runs ###
if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                     for _ in range(args.runs)]
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'amazon2m']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)


### Basic information of datasets ###
n = dataset.graph['num_binodes'] #number of tokens
num_nodes = dataset.graph['num_nodes'] #number of nodes
e = dataset.graph['H'].shape[1] #number of hyperedges
c = max(dataset.label.max().item() + 1, dataset.label.shape[1]) #number of class
d = dataset.graph['node_feat'].shape[1] #number of features


print(f"dataset {args.dataset} | num token {n} | num node {num_nodes} | num edge {e}| num node feats {d} | num classes {c}")

dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
dataset.graph['edge_index_bipart']= dataset.graph['edge_index_bipart'].to(device)
dataset.graph['H']=dataset.graph['H'].to(device)


### Load method ###
model = parse_method(args, n, num_nodes, c, d, e, device)

criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)


adjs = []
adj_bipart, _ = remove_self_loops(dataset.graph['edge_index_bipart'])
adj_bipart, _ = add_self_loops(adj_bipart, num_nodes=dataset.graph['num_binodes'])
adjs.append(adj_bipart)
dataset.graph['adjs'] = adjs#len2

### Training loop ###
for run in tqdm(range(args.runs)):
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        out, link_loss_ = model(args, dataset.graph['node_feat'], dataset.graph['adjs'], dataset.graph['H'], args.tau)

        out = F.log_softmax(out, dim=1)
        loss0 = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
        loss1 = args.lamda * sum(link_loss_) / len(link_loss_)
        if args.sloss:
            loss = loss0 - loss1
        else:
            loss = loss0
        loss.backward()
        optimizer.step()

        if epoch % args.eval_step == 0:
            result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]
                if args.save_model:
                    torch.save(model.state_dict(), args.model_dir + f'{args.dataset}-{args.method}.pkl')

            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
    logger.print_statistics(run)
     
results = logger.print_statistics()
