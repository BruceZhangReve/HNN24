import sys
sys.path.append('/data/lige/HKN')# Please change accordingly!

import math
from numpy import dtype
from sklearn import manifold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from manifolds.poincare import PoincareBall
from manifolds.base import ManifoldParameter
from kernels.poincare_kernel_points import load_kernels


"""
Section 1: Helper Functions
"""
#get dimensions for layers
def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    if args.task in ['lp', 'rec']:
        dims += [args.dim]
        acts += [act]
        n_curvatures = args.num_layers
    else:
        n_curvatures = args.num_layers - 1
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures

#This is different from original one
def gather(x, idx, method=2):
    """
    Used to extract certain rows/ slicing rows: x[idx]
    X[[0,2],[1,1]]:= tensor([matrix(X_row0,X_row2),matrix(X_row1,X_row1)])
    
    Basically 
    Method 0: Direct Indexing
    Pros:
    Simple and straightforward.
    Easy to understand and implement.
    Cons:
    Limited to simple indexing scenarios.
    May require manual handling for complex multi-dimensional indexing.
    
    Method 1: Expanded Indexing
    Pros:
    Can handle more complex indexing cases.
    Automatically aligns tensor shapes using unsqueeze and expand.
    Cons:
    Potentially higher memory consumption due to expanded tensors.
    Slightly more complex than direct indexing.
    
    Method 2: Multi-Step Expansion Indexing
    Pros:
    Provides fine-grained control for very complex multi-dimensional indexing.
    Ensures tensor shapes are properly aligned through multiple unsqueeze and expand steps.
    Cons:
    More complex implementation.
    Potential for higher memory usage and performance overhead due to repeated expansions.
    """

    # Check if idx contains out-of-range indices
    max_index = x.size(0)
    out_of_range = (idx >= max_index) | (idx < 0)

    if out_of_range.any():
        # Create a new row of 0s
        new_row = torch.zeros((1, x.size(1)), dtype=x.dtype, device=x.device)
        # Append the new row to x
        x = torch.cat([x, new_row], dim=0)
        # Clip indices to be within the range
        idx = torch.where(out_of_range, max_index, idx)

    if method == 0:
        return x[idx]
    elif method == 1:
        # Ensure x and idx have compatible shapes
        x_expanded = x.unsqueeze(1).expand(-1, idx.size(1), -1)
        idx_expanded = idx.unsqueeze(2).expand(-1, -1, x.size(1))
        return x_expanded.gather(0, idx_expanded)
    elif method == 2:
        # Expand x to match idx shape
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i + 1)
            new_s = list(x.size())
            new_s[i + 1] = ni
            x = x.expand(new_s)

        # Expand idx to match x shape
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i + n)
            new_s = list(idx.size())
            new_s[i + n] = di
            idx = idx.expand(new_s)

        return x.gather(0, idx)
    else:
        raise ValueError('Unknown method')

def generate_nei_mask(nei):
    """
    Generate a 0/1 adjacency matrix (nei_mask) from the given nei tensor.
    
    Args:
    nei: A tensor where each row contains the indices of nodes connected to the corresponding node.
         -1 is used as a placeholder and should be ignored.
    
    Returns:
    A 0/1 adjacency matrix (nei_mask) representing the connections between nodes.
    """
    num_nodes = nei.size(0)
    nei_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.int32)

    # Create a mask to ignore -1 values
    mask = (nei != -1)
    
    # Use broadcasting to create row and column indices
    row_indices = torch.arange(num_nodes).unsqueeze(1).expand_as(nei)
    col_indices = nei
    
    # Apply the mask to filter out -1 values
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]
    
    # Set the corresponding entries in the nei_mask to 1
    nei_mask[row_indices, col_indices] = 1
    
    return nei_mask


"""
Section 2: Poincare Linears
"""

#BLinear -v2, less operations, solves vanishing gradient issue, but not good performance?

class BLinear(nn.Module):
    def __init__(self, manifold, in_features, out_features, c, dropout, nonlin=None, use_bias=True):
        super(BLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.act = nonlin
        self.E_linear = nn.Linear(in_features, out_features, bias = use_bias).to(torch.float64)

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Xavier 初始化 E_linear 的权重
        nn.init.xavier_uniform_(self.E_linear.weight, gain=nn.init.calculate_gain('relu'))
        if self.E_linear.bias is not None:
            nn.init.constant_(self.E_linear.bias, 0)

    def forward(self, x):
        # Apply dropout to the input
        x = self.dropout(x)

        x_tan = self.manifold.logmap0(x, self.c)
        x_tan = self.E_linear(x_tan)
        if self.act is not None:
            x_tan = self.act(x_tan)

        # Project the result
        res = self.manifold.proj(self.manifold.expmap0(x_tan, c=self.c), c=self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}, use_bias={}, act={}, dropout_rate={}'.format(
            self.in_features, self.out_features, self.c, self.use_bias, self.act, self.dropout_rate)
#BLinear(manifold, in_features, out_features, c=c, dropout=0.5, act=nonlin, use_bias=True)



class BMLP(nn.Module):
    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(BMLP, self).__init__()
        self.linear = BLinear(manifold, in_features, out_features, c, dropout, act, use_bias)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        return h,adj



"""
Section 3: Kernel Point Aggregation
"""
def init_KP(manifold, KP_extent, K, in_channels, c):
    kernel_tangents = load_kernels(manifold,
                                   radius=KP_extent,
                                   num_kpoints=K,
                                   dimension=in_channels,
                                   c=c,
                                   random=False)
    return nn.Parameter(kernel_tangents, requires_grad=False)

def get_kernel_pos(manifold, kernel_tangents, x, c, KP_extent, transp, deformable=False):
    n, d = x.shape
    radius = KP_extent
    K = kernel_tangents.shape[0]

    if not transp:
        res = manifold.expmap0(kernel_tangents, c=c).repeat(n, 1, 1)
    else:
        x_k = x.repeat(1, 1, K - 1).view(n, K - 1, d)
        tmp = manifold.ptransp0(x_k, kernel_tangents[1:], c=c)
        tmp = manifold.proj(tmp, c=c)
        tmp = manifold.expmap(x_k, tmp, c=c)
        res = torch.concat((tmp, x.view(n, 1, d)), 1)

    if deformable:
        pass

    return res

def get_nei_kernel_dis(manifold, x_kernel, x_nei, c):
    n, nei_num, d = x_nei.shape
    feature_points = x_nei.repeat(1, 1, 1, x_kernel.shape[1]).view(n, nei_num, x_kernel.shape[1], d).swapaxes(1, 2)
    kernel_points = x_kernel.repeat(1, 1, 1, nei_num).view(n, x_kernel.shape[1], nei_num, d)
    return torch.sqrt(manifold.sqdist(feature_points, kernel_points, c=c))

def transport_x(manifold, x, x_nei, c):
    x0_nei = manifold.expmap0(manifold.ptransp0back(
        x.repeat(1, 1, x_nei.shape[1]).view(x_nei.shape),
        manifold.logmap(x.repeat(1, 1, x_nei.shape[1]).view(x_nei.shape), x_nei, c=c),
        c=c
    ), c=c)
    x0 = manifold.origin(x.shape[-1], c=c).repeat(x.shape[0], 1)
    return x0, x0_nei

def apply_kernel_transform(manifold, linears, x_nei):
    res = [linears[k](x_nei).unsqueeze(1) for k in range(len(linears))]
    return torch.concat(res, dim=1)

def avg_kernel(manifold, klein_x_nei_transform, x_nei_kernel_dis):
    klein_x_nei_transform = klein_x_nei_transform.swapaxes(1, 2)
    x_nei_kernel_dis = x_nei_kernel_dis.swapaxes(1, 2)
    return manifold.klein_midpoint(klein_x_nei_transform, x_nei_kernel_dis)

def kernel_point_aggregation_forward(manifold, linears, x, nei, nei_mask, kernel_tangents, c, KP_extent, transp=True, sample=False, sample_num=16, deformable=False):
    if sample:
        nei, nei_mask = sample_nei(nei, nei_mask, sample_num)
    
    x_nei = gather(x, nei)
    
    if transp:
        x, x_nei = transport_x(manifold, x, x_nei, c)
    
    n, nei_num, d = x_nei.shape
    kernels = get_kernel_pos(manifold, kernel_tangents, x, c, KP_extent, transp, deformable)
    x_nei_kernel_dis = get_nei_kernel_dis(manifold, kernels, x_nei, c)
    nei_mask = nei_mask.repeat(1, 1, kernels.shape[1]).view(n, kernels.shape[1], nei_num)
    x_nei_kernel_dis = x_nei_kernel_dis * nei_mask
    x_nei_transform = apply_kernel_transform(manifold, linears, x_nei)
    klein_x_nei_transform = manifold.poincare_to_klein(x_nei_transform, c=c)
    klein_x_nei_transform = avg_kernel(manifold, klein_x_nei_transform, x_nei_kernel_dis)
    klein_x_final = manifold.klein_midpoint(klein_x_nei_transform)
    x_final = manifold.klein_to_poincare(klein_x_final, c=c)
    return x_final



