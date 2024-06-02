from gnns import *
from hypergt import *
from data_utils import normalize

def parse_method(args, n,num_nodes, c, d, e, device):
    if args.method == 'hypergt':
        model=HyperGT(n,num_nodes, d, args.hidden_channels, c, e, num_layers=args.num_layers, dropout=args.dropout,
                    num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
                    use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
                    nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    # dataset, protocol
    parser.add_argument('--method', '-m', type=str, default='hypergt')
    parser.add_argument('--sloss', action='store_true', help='use structure regularization')
    parser.add_argument('--pe', type=str, default='none', help='use positional encoding')
    parser.add_argument('--hefeat', type=str, default='none')#rand mean zero load
    parser.add_argument('--feature_noise', type=str, default=0.0)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets with fixed splits, semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20, help='labeled nodes randomly selected')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                        help='evaluation metric')
    parser.add_argument('--knn_num', type=int, default=5, help='number of k for KNN graph')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='../model/')

    # hyper-parameter for model arch and training
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')

    # hyper-parameter for hypergt
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--M', type=int,
                        default=30, help='number of random features')
    parser.add_argument('--use_gumbel', action='store_true', help='use gumbel softmax for message passing')
    parser.add_argument('--use_residual', action='store_true', help='use residual link for each GNN layer')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_act', action='store_true', help='use non-linearity for each layer')
    parser.add_argument('--use_jk', action='store_true', help='concat the layer-wise results in the final layer')
    parser.add_argument('--K', type=int, default=10, help='num of samples for gumbel softmax sampling')
    parser.add_argument('--tau', type=float, default=0.25, help='temperature for gumbel softmax')
    parser.add_argument('--lamda', type=float, default=0.1, help='weight for edge reg loss')
    parser.add_argument('--rb_order', type=int, default=0, help='order for relational bias, 0 for not use')
    parser.add_argument('--rb_trans', type=str, default='sigmoid', choices=['sigmoid', 'identity'],
                        help='non-linearity for relational bias')
    parser.add_argument('--batch_size', type=int, default=10000)



