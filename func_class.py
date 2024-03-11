import torch
from typing import Optional
import math
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.module import Module


def ExtractV2E(edge_index,num_nodes,num_hyperedges):
    # Assume edge_index = [V|E;E|V]
#     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    if not ((num_nodes+num_hyperedges-1) == edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[0].min()  # cidx: [V...|cidx E...]
    V2E = edge_index[:, :cidx].type(torch.LongTensor)
    return V2E

def ConstructH(edge_index_0,num_nodes):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    """
#     ipdb.set_trace()
    edge_index = torch.zeros_like(edge_index_0,dtype=edge_index_0.dtype)
    edge_index[0]=edge_index_0[0]-edge_index_0[0].min()
    edge_index[1]=edge_index_0[1]-edge_index_0[1].min()
    v=torch.ones(edge_index.shape[1])
    # Don't use edge_index[0].max()+1, as some nodes maybe isolated
    num_hyperedges = edge_index[1].max()+1
    H=torch.sparse.FloatTensor(edge_index, v, torch.Size([num_nodes, num_hyperedges]))
    return H

def add_self_loops(edge_index, edge_weight: Optional[torch.Tensor] = None,
                   fill_value: float = 1., num_nodes: Optional[int] = None):
    
    N = num_nodes

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    # if edge_index.min() > 0:
    #     loop_index = loop_index + edge_index.min()

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((N, ), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight

class SparseLinear(Module):
    r"""
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # init.ones_(self.weight)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # wb=torch.sparse.mm(input,self.weight.T).to_dense()+self.bias
        wb=torch.sparse.mm(input,self.weight.T)
        if self.bias is not None:
            out = wb + self.bias
        else:
            out = wb
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
