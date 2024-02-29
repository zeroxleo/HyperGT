# Hypergraph Transformer for Semi-Supervised Classification

This is the repo for our ICASSP 2024 paper: [Hypergraph Transformer for Semi-Supervised Classification](https://arxiv.org/pdf/2312.11385.pdf).

## Recommend Environment:
```
conda create -n "hyperht" python=3.9
conda activate hypergt
pip install -r requirements.txt
```

## Data Preparation:
Create a folder `../data/pyg_data/hypergraph_dataset_updated` and download the datasets from [here]()


You can also set up the following three directories and then unzip the raw data zip file into `p2raw`:
```
p2root: './data/pyg_data/hypergraph_dataset_updated/'
p2raw: './data/AllSet_all_raw_data/'
p2dgl_data: './data/dgl_data_raw/'
```
 The raw data zip file can be found in this [link](https://github.com/jianhao2016/AllSet/tree/main/data/raw_data).

## Acknowledgement

The pipeline for training is developed on basis of the [Nodeformer](https://github.com/qitianwu/NodeFormer) work. The pipeline for data preprocessing is based on the [AllSet](https://github.com/jianhao2016/AllSet) work. Sincere appreciation is extended for their valuable contributions.

## Citation

If you use this code, please cite our paper:

```
@inproceedings{liu2023hypergraph,
  title={Hypergraph Transformer for Semi-Supervised Classification},
  author={Zexi Liu and Bohan Tang and Ziyuan Ye and Xiaowen Dong and Siheng Chen and Yanfeng Wang},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  organization={IEEE}
}
```


