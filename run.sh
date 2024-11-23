python main.py --dataset congress-bills-100 --rand_split --metric acc --method hypergt --lr 1e-3 \
    --weight_decay 5e-2 --num_layers 2 --hidden_channels 156 --num_heads 1 --rb_order 0 --rb_trans sigmoid \
    --lamda 1.0 --M 30 --K 10  --use_bn --use_residual --use_gumbel --runs 10 --epochs 500 --dropout 0.3 \
    --device 3 --data_dir ./data/pyg_data/hypergraph_dataset_updated --sloss --pe HEPEHtEPE \
    --hefeat mean --feature_noise 1.0

python main.py --dataset senate-committees-100 --rand_split --metric acc --method hypergt --lr 1e-3 \
    --weight_decay 5e-2 --num_layers 2 --hidden_channels 64 --num_heads 1 --rb_order 0 --rb_trans sigmoid \
    --lamda 1.0 --M 30 --K 10  --use_bn --use_residual --use_gumbel --runs 10 --epochs 500 --dropout 0.3 \
    --device 2 --data_dir ./data/pyg_data/hypergraph_dataset_updated --sloss --pe HEPEHtEPE \
    --hefeat mean --feature_noise 1.0

python main.py --dataset walmart-trips-100 --rand_split --metric acc --method hypergt --lr 1e-3 \
    --weight_decay 0 --num_layers 3 --hidden_channels 128 --num_heads 1 --rb_order 0 --rb_trans sigmoid \
    --lamda 1.0 --M 30 --K 10  --use_bn --use_residual --use_gumbel --runs 10 --epochs 500 --dropout 0.0 \
    --device 3 --data_dir ./data/pyg_data/hypergraph_dataset_updated --sloss --pe HEPEHtEPE \
    --hefeat mean --feature_noise 1.0   

python main.py --dataset house-committees-100 --rand_split --metric acc --method hypergt --lr 1e-3 \
    --weight_decay 5e-2 --num_layers 2 --hidden_channels 64 --num_heads 1 --rb_order 0 --rb_trans sigmoid \
    --lamda 1.0 --M 30 --K 10  --use_bn --use_residual --use_gumbel --runs 10 --epochs 500 --dropout 0.3 \
    --device 2 --data_dir ./data/pyg_data/hypergraph_dataset_updated --sloss --pe HEPEHtEPE \
    --hefeat mean --feature_noise 1.0
