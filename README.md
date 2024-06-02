# Hypergraph Transformer for Semi-Supervised Classification

The official implementation for "Hypergraph Transformer for Semi-Supervised Classification" which is accepted to ICASSP 2024 as a lecture presentation. [[paper]](https://arxiv.org/pdf/2312.11385.pdf) [[slides]](https://drive.google.com/file/d/1NTQYmLrxRng0Xk0tiqJMephz4Bavvq3w/view?usp=sharing)



## Recommend Environment:
```
conda create -n "hyperht" python=3.9
conda activate hypergt
bash install.sh
```

## Data Preparation:
Create a folder `../data/raw_data` and download the raw datasets from [here](https://huggingface.co/datasets/peihaowang/edgnn-hypergraph-dataset/tree/main).

The directory structure should look like:
```
HyperGT/
  <source code files>
  ...
  data
    raw_data
      congress-bills
      senate-committees
      walmart-trips
      house-committees
```

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


