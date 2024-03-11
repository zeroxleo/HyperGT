import torch
import pickle
import os

import os.path as osp

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset



def save_data_to_pickle(data, p2root = '../data/', file_name = None):
    '''
    if file name not specified, use time stamp.
    '''
#     now = datetime.now()
#     surfix = now.strftime('%b_%d_%Y-%H:%M')
    surfix = 'star_expansion_dataset'
    if file_name is None:
        tmp_data_name = '_'.join(['Hypergraph', surfix])
    else:
        tmp_data_name = file_name
    p2he_StarExpan = osp.join(p2root, tmp_data_name)
    if not osp.isdir(p2root):
        os.makedirs(p2root)
    with open(p2he_StarExpan, 'bw') as f:
        pickle.dump(data, f)
    return p2he_StarExpan


class dataset_Hypergraph(InMemoryDataset):
    def __init__(self, root = './data/pyg_data/hypergraph_dataset_updated/', name = None, 
                 train_percent = 0.01,
                 feature_noise = None,
                 transform=None, pre_transform=None):
        
        existing_dataset = ['walmart-trips-100', 'house-committees-100','senate-committees-100','congress-bills-100']
        if name not in existing_dataset:
            raise ValueError(f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name
        
        self.feature_noise = feature_noise
        self._train_percent = train_percent
        
        if not osp.isdir(root):
            os.makedirs(root)
            
        self.root = root
        self.myraw_dir = osp.join(root, self.name, 'raw')
        self.myprocessed_dir = osp.join(root, self.name, 'processed')
        
        super(dataset_Hypergraph, self).__init__(osp.join(root, name), transform, pre_transform)

        
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent.item()


    @property
    def raw_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'{self.name}_noise_{self.feature_noise}']
        else:
            file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        if self.feature_noise is not None:
            file_names = [f'data_noise_{self.feature_noise}.pt']
        else:
            file_names = ['data.pt']
        return file_names

    @property
    def num_features(self):
        return self.data.num_node_features


    def process(self):
        p2f = osp.join(self.myraw_dir, self.raw_file_names[0])
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
