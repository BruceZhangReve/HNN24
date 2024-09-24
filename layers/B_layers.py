import sys
sys.path.append('/data/lige/HKN')  # Please change accordingly!

import math
from sklearn import manifold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from manifolds.poincare import PoincareBall
from manifolds.base import ManifoldParameter
from kernels.poincare_kernel_points import load_kernels

"""
Section 1: Helper Functions
"""
def empty_cache_on_device(device_id):
    """
    Clear CUDA cache on the specified device.

    Parameters:
    device_id (int): The ID of the CUDA device.
    """
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

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

"""
Section 2: Poincare Linears
"""
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

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(torch.float64))
        self.bias = nn.Parameter(torch.Tensor(out_features).to(torch.float64)) if use_bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training).to(self.c.device)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)

        del mv, drop_weight
        torch.cuda.empty_cache()

        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c).to(self.c.device)
            hyp_bias = self.manifold.proj(self.manifold.expmap0(bias, self.c), self.c)

            del bias
            torch.cuda.empty_cache()

            res = self.manifold.mobius_add(res, hyp_bias, self.c)
            res = self.manifold.proj(res, self.c)

            del hyp_bias
            torch.cuda.empty_cache()

        if self.act is not None:
            res = self.act(self.manifold.logmap0(res, self.c))
            res = self.manifold.proj_tan0(res, self.c)
            res = self.manifold.proj(self.manifold.expmap0(res, self.c), self.c)

        return res

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, c={self.c}, use_bias={self.use_bias}, act={self.act}'

class BAct(nn.Module):
    def __init__(self, manifold, c, act):
        super(BAct, self).__init__()
        self.manifold = manifold
        self.c = c
        self.act = act

    def forward(self, x):
        #Note proj is contained in expmap0
        return self.manifold.expmap0(self.manifold.proj_tan0(self.act(self.manifold.logmap0(x, c=self.c)), c=self.c), c=self.c)

    def extra_repr(self):
        return f'c={self.c}, act={self.act}'

class BMLP(nn.Module):
    def __init__(self, manifold, in_features, out_features, c1, c2, dropout, act, use_bias):
        super(BMLP, self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.linear1 = BLinear(manifold, in_features, out_features, c1, dropout, act, use_bias)
        self.linear2 = BLinear(manifold, out_features, out_features, c2, dropout, None, use_bias)

    def forward(self, x_nei_transform):
        input_device = x_nei_transform.device
        x_nei_transform_additional6 = x_nei_transform.to(self.c1.device)
        x_nei_transform_additional6 = self.linear1(x_nei_transform_additional6)
        h = x_nei_transform_additional6.to(input_device)

        del x_nei_transform
        torch.cuda.empty_cache()
        #print(f"BMLP(cuda6): {torch.cuda.memory_allocated(torch.device('cuda:6')) / (1024**2):.2f} MB allocated, {torch.cuda.memory_reserved(torch.device('cuda:6')) / (1024**2):.2f} MB reserved")
        #print(f"BMLP(cuda5): {torch.cuda.memory_allocated(torch.device('cuda:5')) / (1024**2):.2f} MB allocated, {torch.cuda.memory_reserved(torch.device('cuda:5')) / (1024**2):.2f} MB reserved")

        h_additional5 = h.to(self.c2.device)
        h_additional5 = self.linear2(h_additional5)
        h = h_additional5.to(input_device)

        return h

"""
Section 3: Kernel Point Aggregation
"""

class KernelPointAggregation(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, KP_extent, manifold, use_bias, dropout, c, nonlin=None, aggregation_mode='sum', deformable=False, AggKlein=True, corr=1, nei_agg=2):
        super(KernelPointAggregation, self).__init__()

        self.manifold = manifold
        self.c = c
        self.K = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KP_extent = KP_extent
        self.deformable = deformable
        self.AggKlein = AggKlein
        self.corr = corr
        self.nei_agg = nei_agg

        self.additional_devices = False #Please change settings here, this is not a part of config!!!
        if self.additional_devices:
            self.c_additional6 = self.c.to(torch.device('cuda:6')) # for MLP(linear1)
            self.c_additional5 = self.c.to(torch.device('cuda:5')) # for MLP(linear2)
            self.c_additional4 = self.c.to(torch.device('cuda:3')) # for apply_kernel_transform
            self.c_additional3 = self.c.to(torch.device('cuda:2')) # for avg_kernels
        else:
            self.c_additional6 = self.c # for MLP(linear1)
            self.c_additional5 = self.c # for MLP(linear2)
            self.c_additional4 = self.c # for apply_kernel_transform
            self.c_additional3 = self.c # for avg_kernels/transportx

        if self.corr in [0, 1]:
            self.linears = nn.ModuleList([BLinear(manifold, in_channels, out_channels, self.c_additional4, dropout, None, use_bias) for _ in range(self.K)])
        elif self.corr == 2:
            self.single_linear = BLinear(manifold, in_channels, out_channels, self.c, dropout, nonlin, use_bias)
        else:
            raise NotImplementedError("The specified correlation type is not implemented.")

        if self.nei_agg == 0:
            self.act = BAct(manifold, self.c, nonlin)
        elif self.nei_agg == 1:
            self.atten1 = BLinear(manifold, out_channels + out_channels, 1, self.c, dropout, nonlin=None, use_bias=True)
            self.atten2 = BLinear(manifold, out_channels + out_channels, 1, self.c, dropout, nonlin=None, use_bias=True)
        elif self.nei_agg == 2:
            #First MLP is put on self.c_additional.device
            self.MLP_f = BMLP(manifold, out_channels, 2 * out_channels, self.c_additional6, self.c_additional5, dropout, nonlin, use_bias)
            #Second MLP is put on self.c_additional2.device
            self.MLP_fi = BMLP(manifold, 2 * out_channels, out_channels, self.c_additional6, self.c_additional5, dropout, nonlin, use_bias)
        else:
            raise NotImplementedError("The specified neighbor aggregation type is not implemented.")

        self.kernel_tangents = self.init_KP()

    def init_KP(self):
        return nn.Parameter(load_kernels(self.manifold, radius=self.KP_extent, num_kpoints=self.K, dimension=self.in_channels, c=self.c, random=False), requires_grad=False)

    def get_kernel_pos(self, x, nei, nei_mask, sample, sample_num, transp, radius=None):
        n, d = x.shape
        radius = self.KP_extent if radius is None else radius

        K = self.kernel_tangents.shape[0]
        if not transp:
            res = self.manifold.expmap0(self.kernel_tangents, c=self.c).repeat(n, 1, 1)
        else:
            x_k = x.repeat(1, 1, K - 1).view(n, K - 1, d)
            tmp = self.manifold.ptransp0(x_k, self.kernel_tangents[1:], c=self.c)
            tmp = self.manifold.proj(tmp, c=self.c)
            tmp = self.manifold.expmap(x_k, tmp, c=self.c)
            res = torch.concat((tmp, x.view(n, 1, d)), 1)
        return res

    def get_nei_kernel_dis(self, x_kernel, x_nei):
        if x_nei.dim() == 3:
            #x_nei: The result from "gather()". (n,nei_num,d)
            #x_kernel: The result from "get_kernel_pos()", with kernels defined for each cases. (n,K,nei_num)
            n, nei_num, d = x_nei.shape
            kernel_points = x_kernel.repeat(1,1,1,nei_num).view(n,self.K,nei_num,d)#shape=[n,K,nei_num,d]
            feature_points = x_nei.repeat(1, 1, 1,self.K).view(n, nei_num, self.K, d).swapaxes(1, 2)#shape=[n,K,nei_num,d]
            return torch.sqrt(self.manifold.sqdist(feature_points,kernel_points,c=self.c))#shape=[n,K,nei_num] #Poincare distance
        else:
            #x_nei==X_nei_transform := (n,K,nei_num,d)
            #x_kernel: The result from "get_kernel_pos()", with kernels defined for each cases. (n,K,nei_num)
            n, K, nei_num, d = x_nei.shape
            kernel_points = x_kernel.repeat(1,1,1,nei_num).view(n,self.K,nei_num,d)#shape=[n,K,nei_num,d]
            feature_points = x_nei #(n,K,nei_num,d)
            return torch.sqrt(self.manifold.sqdist(feature_points,kernel_points,c=self.c))#shape=[n,K,nei_num] #Poincare distance

    def transport_x(self, x, x_nei):
        x_additional = x.to(self.c_additional3.device)
        x_nei_addigtional = x_nei.to(self.c_additional3.device)

        x0_nei_additional = self.manifold.expmap0(
            self.manifold.ptransp0back(x_additional.repeat(1, 1, x_nei_addigtional.shape[1]).view(x_nei_addigtional.shape),
            self.manifold.logmap(x_additional.repeat(1, 1, x_nei_addigtional.shape[1]).view(x_nei_addigtional.shape), x_nei_addigtional, self.c_additional3), self.c_additional3), self.c_additional3)

        x0_additional = self.manifold.origin(x_additional.shape[-1], c=self.c_additional3).repeat(x_additional.shape[0], 1)

        x0_nei = x0_nei_additional.to(self.c.device)
        x0 = x0_additional.to(self.c.device)
        return x0, x0_nei

    def apply_kernel_transform(self, x_nei):
        res = []
        for k in range(self.K):
            transformed = self.linears[k](x_nei).unsqueeze(1)
            res.append(transformed.cpu())
            del transformed
            torch.cuda.empty_cache()
        res = torch.cat(res, dim=1)
        res = res.to(x_nei.device)
        torch.cuda.empty_cache()
        return res

    def avg_kernel(self, x_nei_transform, x_nei_kernel_dis, AggKlein):
        x_nei_transform_additional = x_nei_transform.to(self.c_additional3.device)
        x_nei_kernel_dis_additional = x_nei_kernel_dis.to(self.c_additional3.device)

        x_nei_transform_additional = x_nei_transform_additional.swapaxes(1, 2)
        x_nei_kernel_dis_additional = x_nei_kernel_dis_additional.swapaxes(1, 2)

        if self.corr == 1:
            x_nei_kernel_dis_additional = F.softmax(x_nei_kernel_dis_additional, dim=-1)

        if self.AggKlein == True:
            res = self.manifold.klein_midpoint(x_nei_transform_additional, x_nei_kernel_dis_additional)
            #res = self.manifold.proj(self.manifold.klein_to_poincare(res, self.c_additional3), self.c_additional3)
            return res.to(self.c.device)
        else:
            res = self.manifold.hyperboloid_centroid(x_nei_transform_additional, self.c_additional3, x_nei_kernel_dis_additional)
            #res = self.manifold.proj(self.manifold.hyperboloid_to_poincare(res, self.c_additional3), self.c_additional3)
            return res.to(self.c.device)

    def forward(self, x, nei, nei_mask, transp=True, sample=False, sample_num=16):
        #print(f"before_everything(cuda7): {torch.cuda.memory_allocated(torch.device('cuda:7')) / (1024**2):.2f} MB allocated, {torch.cuda.memory_reserved(torch.device('cuda:7')) / (1024**2):.2f} MB reserved")
        if sample:
            pass
            #nei, nei_mask = self.sample_nei(nei, nei_mask, sample_num)

        x_nei = gather(x, nei)

        if transp:
            x, x_nei = self.transport_x(x, x_nei)

        n, nei_num, d = x_nei.shape
        kernels = self.get_kernel_pos(x, nei, nei_mask, sample, sample_num, transp=not transp)

        if self.corr in [0, 2]:
            x_nei_kernel_dis = self.get_nei_kernel_dis(kernels, x_nei)
            nei_mask = nei_mask.repeat(1, 1, self.K).view(n, self.K, nei_num)
            x_nei_kernel_dis = x_nei_kernel_dis * nei_mask
            if self.corr == 2:
                pass
            else:
                x_nei_transform = self.apply_kernel_transform(x_nei.to(self.c_additional4.device))
                x_nei_transform = x_nei_transform.to(self.c.device)
        else:
            x_nei_transform = self.apply_kernel_transform(x_nei.to(self.c_additional4.device))
            x_nei_transform = x_nei_transform.to(self.c.device)
            if x_nei.shape[-1] != x_nei_transform.shape[-1]:
                raise ValueError("Don't change dimension in linear transformation step if use corr==1")
            x_nei_kernel_dis = self.get_nei_kernel_dis(kernels, x_nei_transform)
            nei_mask = nei_mask.repeat(1, 1, self.K).view(n, self.K, nei_num)
            x_nei_kernel_dis = x_nei_kernel_dis * nei_mask


        #print(f"before(cuda7): {torch.cuda.memory_allocated(torch.device('cuda:7')) / (1024**2):.2f} MB allocated, {torch.cuda.memory_reserved(torch.device('cuda:7')) / (1024**2):.2f} MB reserved")
        del kernels, nei_mask
        #print(f"before(cuda7): {torch.cuda.memory_allocated(torch.device('cuda:7')) / (1024**2):.2f} MB allocated, {torch.cuda.memory_reserved(torch.device('cuda:7')) / (1024**2):.2f} MB reserved")
        empty_cache_on_device(self.c.device)
        #print(f"after(cuda7): {torch.cuda.memory_allocated(torch.device('cuda:7')) / (1024**2):.2f} MB allocated, {torch.cuda.memory_reserved(torch.device('cuda:7')) / (1024**2):.2f} MB reserved")

        if self.corr == 2:
            if self.AggKlein:
                klein_x_nei = self.manifold.poincare_to_klein(x_nei, self.c)
                klein_x_nei = self.manifold.klein_proj(klein_x_nei, self.c)
                klein_x_nei = self.avg_kernel(klein_x_nei, x_nei_kernel_dis, self.AggKlein)
                poincare_x_nei = self.manifold.klein_to_poincare(klein_x_nei, self.c)
                poincare_x_nei_transform = self.single_linear(poincare_x_nei)
                poincare_x_nei_transform = self.manifold.proj(poincare_x_nei_transform, self.c)
                klein_x_nei_transform = self.manifold.poincare_to_klein(poincare_x_nei_transform, self.c)
                klein_x_nei_transform = self.manifold.klein_proj(klein_x_nei_transform, self.c)
                klein_x_final = self.manifold.klein_midpoint(klein_x_nei_transform)
                x_final = self.manifold.klein_to_poincare(klein_x_final, self.c)
                x_final = self.manifold.proj(x_final, self.c)
            else:
                hyperboloid_x_nei = self.manifold.poincare_to_hyperboloid(x_nei, self.c)
                hyperboloid_x_nei = self.manifold.hyperboloid_proj(hyperboloid_x_nei, self.c)
                hyperboloid_x_nei = self.avg_kernel(hyperboloid_x_nei, x_nei_kernel_dis, not self.AggKlein)
                poincare_x_nei = self.manifold.hyperboloid_to_poincare(hyperboloid_x_nei, self.c)
                poincare_x_nei_transform = self.single_linear(poincare_x_nei)
                poincare_x_nei_transform = self.manifold.proj(poincare_x_nei_transform, self.c)
                hyperboloid_x_nei_transform = self.manifold.poincare_to_hyperboloid(poincare_x_nei_transform, self.c)
                hyperboloid_x_nei_transform = self.manifold.hyperboloid_proj(hyperboloid_x_nei_transform, self.c)
                hyperboloid_x_final = self.manifold.hyperboloid_centroid(hyperboloid_x_nei_transform, self.c)
                x_final = self.manifold.hyperboloid_to_poincare(hyperboloid_x_final, self.c)
                x_final = self.manifold.proj(x_final, self.c)
            x_final = self.act(x_final)
            return self.manifold.proj(x_final, self.c)

        elif self.corr in [0, 1]:
            if self.AggKlein:
                klein_x_nei_transform = self.manifold.klein_proj(self.manifold.poincare_to_klein(x_nei_transform, self.c), self.c)
                del x_nei_transform
                torch.cuda.empty_cache()

                klein_x_nei_transform = self.avg_kernel(klein_x_nei_transform, x_nei_kernel_dis, self.AggKlein)
                #klein_x_nei_transform_additional = klein_x_nei_transform.to(self.c_additional3.device)
                #klein_x_nei_transform_additional = self.manifold.klein_proj(klein_x_nei_transform_additional, self.c_additional3)
                #x_nei_kernel_dis_additional = x_nei_kernel_dis.to(self.c_additional3.device)
                #klein_x_nei_transform_additional = self.avg_kernel(klein_x_nei_transform_additional, x_nei_kernel_dis_additional, self.AggKlein)
                #klein_x_nei_transform_additional = self.manifold.klein_proj(klein_x_nei_transform_additional, self.c_additional3)
                #klein_x_nei_transform = klein_x_nei_transform_additional.to(self.c.device)

                if self.nei_agg == 0:
                    klein_x_final = self.manifold.klein_midpoint(klein_x_nei_transform)
                    x_final = self.manifold.klein_to_poincare(klein_x_final, self.c)
                    x_final = self.manifold.proj(x_final, self.c)
                    x_final = self.act(x_final)
                    return x_final
                elif self.nei_agg == 1:
                    x_nei_transform = self.manifold.klein_to_poincare(klein_x_nei_transform, self.c)
                    x_nei_transform = self.manifold.proj(poincare_x_nei_transform, self.c)

                    attention1 = F.softmax(self.atten1(torch.cat((x_nei_transform, torch.zeros_like(x_nei_transform)), dim=-1)).squeeze(-1), dim=-1)
                    attention2 = F.softmax(self.atten2(torch.cat((x_nei_transform, torch.zeros_like(x_nei_transform)), dim=-1)).squeeze(-1), dim=-1)
                    multihead_attention = (attention1 + attention2) / 2

                    klein_x_nei_transform = self.manifold.poincare_to_klein(x_nei_transform, self.c)
                    klein_x_nei_transform = self.manifold.klein_proj(klein_x_nei_transform, self.c)
                    klein_x_final = self.manifold.klein_midpoint(klein_x_nei_transform, multihead_attention)
                    x_final = self.manifold.klein_to_poincare(klein_x_final, self.c)
                    x_final = self.manifold.proj(x_final, self.c)
                    x_final = self.act(x_final)
                    return x_final
                elif self.nei_agg == 2:
                    x_nei_transform = self.manifold.klein_to_poincare(klein_x_nei_transform, self.c)
                    x_nei_transform = self.manifold.proj(x_nei_transform, self.c)
                    del klein_x_nei_transform
                    empty_cache_on_device(self.c.device)

                    x_nei_transform = self.MLP_f(x_nei_transform)

                    klein_x_final = self.manifold.klein_midpoint(self.manifold.klein_proj(self.manifold.poincare_to_klein(x_nei_transform, self.c), self.c))

                    del x_nei_transform
                    empty_cache_on_device(self.c.device)

                    x_final = self.manifold.proj(self.manifold.klein_to_poincare(klein_x_final, self.c), self.c)

                    del klein_x_final
                    empty_cache_on_device(self.c.device)

                    x_final = self.MLP_fi(x_final)
                    #print(f"having x_final,(cuda7): {torch.cuda.memory_allocated(torch.device('cuda:7')) / (1024**2):.2f} MB allocated, {torch.cuda.memory_reserved(torch.device('cuda:7')) / (1024**2):.2f} MB reserved")

                    return x_final
                else:
                    raise NotImplementedError("The specified neighbor aggregation type is not implemented.")
            else:
                hyperboloid_x_nei_transform = self.manifold.poincare_to_hyperboloid(x_nei_transform, self.c)
                hyperboloid_x_nei_transform = self.manifold.hyperboloid_proj(hyperboloid_x_nei_transform, self.c)
                hyperboloid_x_nei_transform = self.avg_kernel(hyperboloid_x_nei_transform, x_nei_kernel_dis, self.AggKlein)

                if self.nei_agg == 0:
                    hyperboloid_x_final = self.manifold.hyperboloid_centroid(hyperboloid_x_nei_transform, self.c)
                    hyperboloid_x_final = self.manifold.hyperboloid_proj(hyperboloid_x_final, self.c)
                    x_final = self.manifold.hyperboloid_to_poincare(hyperboloid_x_final, self.c)
                    x_final = self.manifold.proj(x_final, self.c)
                    x_final = self.act(x_final)
                    return self.manifold.proj(x_final, self.c)
                elif self.nei_agg == 1:
                    poincare_x_nei_transform = self.manifold.hyperboloid_to_poincare(hyperboloid_x_nei_transform, self.c)
                    poincare_x_nei_transform = self.manifold.proj(poincare_x_nei_transform, self.c)

                    attention1 = F.softmax(self.atten1(torch.cat((poincare_x_nei_transform, torch.zeros_like(poincare_x_nei_transform)), dim=-1)).squeeze(-1), dim=-1)
                    attention2 = F.softmax(self.atten2(torch.cat((poincare_x_nei_transform, torch.zeros_like(poincare_x_nei_transform)), dim=-1)).squeeze(-1), dim=-1)
                    multihead_attention = (attention1 + attention2) / 2

                    hyperboloid_x_nei_transform = self.manifold.poincare_to_hyperboloid(poincare_x_nei_transform, self.c)
                    hyperboloid_x_nei_transform = self.manifold.hyperboloid_proj(hyperboloid_x_nei_transform, self.c)
                    hyperboloid_x_final = self.manifold.hyperboloid_centroid(hyperboloid_x_nei_transform, multihead_attention)
                    hyperboloid_x_final = self.manifold.hyperboloid_proj(hyperboloid_x_final, self.c)
                    x_final = self.manifold.hyperboloid_to_poincare(hyperboloid_x_final, self.c)
                    x_final = self.manifold.proj(x_final, self.c)
                    x_final = self.act(x_final)
                    return self.manifold.proj(x_final, self.c)
                elif self.nei_agg == 2:
                    poincare_x_nei_transform = self.manifold.hyperboloid_to_poincare(hyperboloid_x_nei_transform, self.c)
                    poincare_x_nei_transform = self.manifold.proj(poincare_x_nei_transform, self.c)
                    hyperboloid_x_final = self.manifold.hyperboloid_proj(self.manifold.hyperboloid_centroid(self.manifold.hyperboloid_proj(self.manifold.poincare_to_hyperboloid(self.manifold.proj(self.MLP_f(poincare_x_nei_transform), self.c), self.c), self.c), self.c), self.c)
                    x_final = self.manifold.proj(self.MLP_fi(self.manifold.proj(self.manifold.hyperboloid_to_poincare(hyperboloid_x_final, self.c), self.c)), c=self.c)
                    return x_final
                else:
                    raise NotImplementedError("The specified neighbor aggregation type is not implemented.")
        else:
            raise NotImplementedError("The specified correlation type is not implemented.")

class KPGraphConvolution(nn.Module):
    """
    Hyperbolic Kernel Point Convolution Layer.
    """
    def __init__(self, manifold, kernel_size, KP_extent, in_features, out_features, use_bias, dropout, c, nonlin, deformable, AggKlein, corr, nei_agg):
        super(KPGraphConvolution, self).__init__()
        self.net = KernelPointAggregation(kernel_size, in_features, out_features, KP_extent, manifold, use_bias, dropout, c, nonlin, aggregation_mode='sum', deformable=deformable, AggKlein=AggKlein, corr=corr, nei_agg=nei_agg)

    def forward(self, input):
        x, nei, nei_mask = input
        h = self.net(x, nei, nei_mask)
        return h, nei, nei_mask

"""
#My old version
class KernelPointAggregation(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, KP_extent, manifold, use_bias, dropout, c, nonlin=None, aggregation_mode='sum', deformable=False, AggKlein=True, corr=1, nei_agg=2):
        super(KernelPointAggregation, self).__init__()

        self.manifold = manifold
        self.c = c
        self.K = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KP_extent = KP_extent
        self.deformable = deformable
        self.AggKlein = AggKlein
        self.corr = corr
        self.nei_agg = nei_agg

        if True:
            #slect to ask help from other devices, need to add a hyperparameter later!!!
            self.c_additional6 = self.c.to(torch.device('cuda:6')) # for MLP(linear1)
            self.c_additional5 = self.c.to(torch.device('cuda:5')) # for MLP(linear2)
            self.c_additional4 = self.c.to(torch.device('cuda:4')) # for apply_kernel_transform
            self.c_additional3 = self.c.to(torch.device('cuda:3')) # for avg_kernels
        else:
            self.c_additional6 = self.c # for MLP(linear1)
            self.c_additional5 = self.c # for MLP(linear2)
            self.c_additional4 = self.c # for apply_kernel_transform
            self.c_additional3 = self.c # for avg_kernels

        if self.corr in [0, 1]:
            self.linears = nn.ModuleList([BLinear(manifold, in_channels, out_channels, self.c_additional4, dropout, None, use_bias) for _ in range(self.K)])
        elif self.corr == 2:
            self.single_linear = BLinear(manifold, in_channels, out_channels, self.c, dropout, nonlin, use_bias)
        else:
            raise NotImplementedError("The specified correlation type is not implemented.")

        if self.nei_agg == 0:
            self.act = BAct(manifold, self.c, nonlin)
        elif self.nei_agg == 1:
            self.atten1 = BLinear(manifold, out_channels + out_channels, 1, self.c, dropout, nonlin=None, use_bias=True)
            self.atten2 = BLinear(manifold, out_channels + out_channels, 1, self.c, dropout, nonlin=None, use_bias=True)
        elif self.nei_agg == 2:
            #First MLP is put on self.c_additional.device
            self.MLP_f = BMLP(manifold, out_channels, 2 * out_channels, self.c_additional6, self.c_additional5, dropout, nonlin, use_bias)
            #Second MLP is put on self.c_additional2.device
            self.MLP_fi = BMLP(manifold, 2 * out_channels, out_channels, self.c_additional6, self.c_additional5, dropout, nonlin, use_bias)
        else:
            raise NotImplementedError("The specified neighbor aggregation type is not implemented.")

        self.kernel_tangents = self.init_KP()

    def init_KP(self):
        return nn.Parameter(load_kernels(self.manifold, radius=self.KP_extent, num_kpoints=self.K, dimension=self.in_channels, c=self.c, random=False), requires_grad=False)

    def get_kernel_pos(self, x, nei, nei_mask, sample, sample_num, transp, radius=None):
        n, d = x.shape
        radius = self.KP_extent if radius is None else radius

        K = self.kernel_tangents.shape[0]
        if not transp:
            res = self.manifold.expmap0(self.kernel_tangents, c=self.c).repeat(n, 1, 1)
        else:
            x_k = x.repeat(1, 1, K - 1).view(n, K - 1, d)
            tmp = self.manifold.ptransp0(x_k, self.kernel_tangents[1:], c=self.c)
            tmp = self.manifold.proj(tmp, c=self.c)
            tmp = self.manifold.expmap(x_k, tmp, c=self.c)
            res = torch.concat((tmp, x.view(n, 1, d)), 1)
        return res

    def get_nei_kernel_dis(self, x_kernel, x_nei):
        if x_nei.dim() == 3:
            #x_nei: The result from "gather()". (n,nei_num,d)
            #x_kernel: The result from "get_kernel_pos()", with kernels defined for each cases. (n,K,nei_num)
            n, nei_num, d = x_nei.shape
            kernel_points = x_kernel.repeat(1,1,1,nei_num).view(n,self.K,nei_num,d)#shape=[n,K,nei_num,d]
            feature_points = x_nei.repeat(1, 1, 1,self.K).view(n, nei_num, self.K, d).swapaxes(1, 2)#shape=[n,K,nei_num,d]
            return torch.sqrt(self.manifold.sqdist(feature_points,kernel_points,c=self.c))#shape=[n,K,nei_num] #Poincare distance
        else:
            #x_nei==X_nei_transform := (n,K,nei_num,d)
            #x_kernel: The result from "get_kernel_pos()", with kernels defined for each cases. (n,K,nei_num)
            n, K, nei_num, d = x_nei.shape
            kernel_points = x_kernel.repeat(1,1,1,nei_num).view(n,self.K,nei_num,d)#shape=[n,K,nei_num,d]
            feature_points = x_nei #(n,K,nei_num,d)
            return torch.sqrt(self.manifold.sqdist(feature_points,kernel_points,c=self.c))#shape=[n,K,nei_num] #Poincare distance

    def transport_x(self, x, x_nei):
        x0_nei = self.manifold.expmap0(self.manifold.ptransp0back(x.repeat(1, 1, x_nei.shape[1]).view(x_nei.shape), self.manifold.logmap(x.repeat(1, 1, x_nei.shape[1]).view(x_nei.shape), x_nei, c=self.c), c=self.c), c=self.c)
        x0 = self.manifold.origin(x.shape[-1], c=self.c).repeat(x.shape[0], 1)
        return x0, x0_nei

    def apply_kernel_transform(self, x_nei):
        res = []
        for k in range(self.K):
            transformed = self.linears[k](x_nei).unsqueeze(1)
            res.append(transformed.cpu())
            del transformed
            torch.cuda.empty_cache()
        res = torch.cat(res, dim=1)
        res = res.to(x_nei.device)
        torch.cuda.empty_cache()
        return res

    def avg_kernel(self, x_nei_transform, x_nei_kernel_dis, AggKlein):
        x_nei_transform = x_nei_transform.swapaxes(1, 2)
        x_nei_kernel_dis = x_nei_kernel_dis.swapaxes(1, 2)

        if self.corr == 1:
            x_nei_kernel_dis = F.softmax(x_nei_kernel_dis, dim=-1)

        if AggKlein:
            return self.manifold.klein_midpoint(x_nei_transform, x_nei_kernel_dis)
        else:
            return self.manifold.hyperboloid_centroid(x_nei_transform, self.c_additional3, x_nei_kernel_dis)

    def forward(self, x, nei, nei_mask, transp=True, sample=False, sample_num=16):
        if sample:
            pass
            #nei, nei_mask = self.sample_nei(nei, nei_mask, sample_num)

        x_nei = gather(x, nei)

        if transp:
            x, x_nei = self.transport_x(x, x_nei)

        n, nei_num, d = x_nei.shape
        kernels = self.get_kernel_pos(x, nei, nei_mask, sample, sample_num, transp=not transp)

        if self.corr in [0, 2]:
            x_nei_kernel_dis = self.get_nei_kernel_dis(kernels, x_nei)
            nei_mask = nei_mask.repeat(1, 1, self.K).view(n, self.K, nei_num)
            x_nei_kernel_dis = x_nei_kernel_dis * nei_mask
            if self.corr == 2:
                pass
            else:
                x_nei_transform = self.apply_kernel_transform(x_nei.to(self.c_additional4.device))
                x_nei_transform = x_nei_transform.to(self.c.device)
        else:
            x_nei_transform = self.apply_kernel_transform(x_nei.to(self.c_additional4.device))
            x_nei_transform = x_nei_transform.to(self.c.device)
            if x_nei.shape[-1] != x_nei_transform.shape[-1]:
                raise ValueError("Don't change dimension in linear transformation step if use corr==1")
            x_nei_kernel_dis = self.get_nei_kernel_dis(kernels, x_nei_transform)
            nei_mask = nei_mask.repeat(1, 1, self.K).view(n, self.K, nei_num)
            x_nei_kernel_dis = x_nei_kernel_dis * nei_mask

        del kernels, nei_mask
        torch.cuda.empty_cache()

        if self.corr == 2:
            if self.AggKlein:
                klein_x_nei = self.manifold.poincare_to_klein(x_nei, self.c)
                klein_x_nei = self.manifold.klein_proj(klein_x_nei, self.c)
                klein_x_nei = self.avg_kernel(klein_x_nei, x_nei_kernel_dis, self.AggKlein)
                poincare_x_nei = self.manifold.klein_to_poincare(klein_x_nei, self.c)
                poincare_x_nei_transform = self.single_linear(poincare_x_nei)
                poincare_x_nei_transform = self.manifold.proj(poincare_x_nei_transform, self.c)
                klein_x_nei_transform = self.manifold.poincare_to_klein(poincare_x_nei_transform, self.c)
                klein_x_nei_transform = self.manifold.klein_proj(klein_x_nei_transform, self.c)
                klein_x_final = self.manifold.klein_midpoint(klein_x_nei_transform)
                x_final = self.manifold.klein_to_poincare(klein_x_final, self.c)
                x_final = self.manifold.proj(x_final, self.c)
            else:
                hyperboloid_x_nei = self.manifold.poincare_to_hyperboloid(x_nei, self.c)
                hyperboloid_x_nei = self.manifold.hyperboloid_proj(hyperboloid_x_nei, self.c)
                hyperboloid_x_nei = self.avg_kernel(hyperboloid_x_nei, x_nei_kernel_dis, not self.AggKlein)
                poincare_x_nei = self.manifold.hyperboloid_to_poincare(hyperboloid_x_nei, self.c)
                poincare_x_nei_transform = self.single_linear(poincare_x_nei)
                poincare_x_nei_transform = self.manifold.proj(poincare_x_nei_transform, self.c)
                hyperboloid_x_nei_transform = self.manifold.poincare_to_hyperboloid(poincare_x_nei_transform, self.c)
                hyperboloid_x_nei_transform = self.manifold.hyperboloid_proj(hyperboloid_x_nei_transform, self.c)
                hyperboloid_x_final = self.manifold.hyperboloid_centroid(hyperboloid_x_nei_transform, self.c)
                x_final = self.manifold.hyperboloid_to_poincare(hyperboloid_x_final, self.c)
                x_final = self.manifold.proj(x_final, self.c)
            x_final = self.act(x_final)
            return self.manifold.proj(x_final, self.c)
        elif self.corr in [0, 1]:
            if self.AggKlein:
                klein_x_nei_transform = self.manifold.poincare_to_klein(x_nei_transform, self.c)
                del x_nei_transform
                torch.cuda.empty_cache()

                klein_x_nei_transform_additional = klein_x_nei_transform.to(self.c_additional3.device)
                klein_x_nei_transform_additional = self.manifold.klein_proj(klein_x_nei_transform_additional, self.c_additional3)
                x_nei_kernel_dis_additional = x_nei_kernel_dis.to(self.c_additional3.device)
                klein_x_nei_transform_additional = self.avg_kernel(klein_x_nei_transform_additional, x_nei_kernel_dis_additional, self.AggKlein)
                klein_x_nei_transform_additional = self.manifold.klein_proj(klein_x_nei_transform_additional, self.c_additional3)
                klein_x_nei_transform = klein_x_nei_transform_additional.to(self.c.device)

                if self.nei_agg == 0:
                    klein_x_final = self.manifold.klein_midpoint(klein_x_nei_transform)
                    klein_x_final = self.manifold.klein_proj(klein_x_final, self.c)
                    x_final = self.manifold.klein_to_poincare(klein_x_final, self.c)
                    x_final = self.manifold.proj(x_final, self.c)
                    x_final = self.act(x_final)
                    return self.manifold.proj(x_final, self.c)
                elif self.nei_agg == 1:
                    poincare_x_nei_transform = self.manifold.klein_to_poincare(klein_x_nei_transform, self.c)
                    poincare_x_nei_transform = self.manifold.proj(poincare_x_nei_transform, self.c)

                    attention1 = F.softmax(self.atten1(torch.cat((poincare_x_nei_transform, torch.zeros_like(poincare_x_nei_transform)), dim=-1)).squeeze(-1), dim=-1)
                    attention2 = F.softmax(self.atten2(torch.cat((poincare_x_nei_transform, torch.zeros_like(poincare_x_nei_transform)), dim=-1)).squeeze(-1), dim=-1)
                    multihead_attention = (attention1 + attention2) / 2

                    klein_x_nei_transform = self.manifold.poincare_to_klein(poincare_x_nei_transform, self.c)
                    klein_x_nei_transform = self.manifold.klein_proj(klein_x_nei_transform, self.c)
                    klein_x_final = self.manifold.klein_midpoint(klein_x_nei_transform, multihead_attention)
                    klein_x_final = self.manifold.klein_proj(klein_x_final, self.c)
                    x_final = self.manifold.klein_to_poincare(klein_x_final, self.c)
                    x_final = self.manifold.proj(x_final, self.c)
                    x_final = self.act(x_final)
                    return self.manifold.proj(x_final, self.c)
                elif self.nei_agg == 2:
                    x_nei_transform = self.manifold.klein_to_poincare(klein_x_nei_transform, self.c)
                    del klein_x_nei_transform
                    torch.cuda.empty_cache()

                    x_nei_transform = self.manifold.proj(x_nei_transform, self.c)
                    #x_nei_transform_additional = x_nei_transform.to(self.c_additional6.device)
                    #x_nei_transform_additional = self.MLP_f(x_nei_transform_additional)
                    #x_nei_transform = x_nei_transform_additional.to(self.c.device)
                    x_nei_transform = self.MLP_f(x_nei_transform)

                    klein_x_final = self.manifold.klein_proj(self.manifold.klein_midpoint(self.manifold.klein_proj(self.manifold.poincare_to_klein(self.manifold.proj(x_nei_transform, self.c), self.c), self.c)), self.c)

                    del x_nei_transform
                    torch.cuda.empty_cache()

                    x_final = self.manifold.proj(self.manifold.klein_to_poincare(klein_x_final, self.c), self.c)

                    del klein_x_final
                    torch.cuda.empty_cache()

                    #x_final_additional2 = x_final.to(self.c_additional5.device)
                    #x_final_additional2 = self.MLP_fi(x_final_additional2)
                    #x_final = x_final_additional2.to(self.c.device)
                    x_final = self.MLP_fi(x_final)

                    x_final = self.manifold.proj(x_final, self.c)
                    return x_final
                else:
                    raise NotImplementedError("The specified neighbor aggregation type is not implemented.")
            else:
                hyperboloid_x_nei_transform = self.manifold.poincare_to_hyperboloid(x_nei_transform, self.c)
                hyperboloid_x_nei_transform = self.manifold.hyperboloid_proj(hyperboloid_x_nei_transform, self.c)
                hyperboloid_x_nei_transform = self.avg_kernel(hyperboloid_x_nei_transform, x_nei_kernel_dis, self.AggKlein)
                hyperboloid_x_nei_transform = self.manifold.hyperboloid_proj(hyperboloid_x_nei_transform, self.c)

                if self.nei_agg == 0:
                    hyperboloid_x_final = self.manifold.hyperboloid_centroid(hyperboloid_x_nei_transform, self.c)
                    hyperboloid_x_final = self.manifold.hyperboloid_proj(hyperboloid_x_final, self.c)
                    x_final = self.manifold.hyperboloid_to_poincare(hyperboloid_x_final, self.c)
                    x_final = self.manifold.proj(x_final, self.c)
                    x_final = self.act(x_final)
                    return self.manifold.proj(x_final, self.c)
                elif self.nei_agg == 1:
                    poincare_x_nei_transform = self.manifold.hyperboloid_to_poincare(hyperboloid_x_nei_transform, self.c)
                    poincare_x_nei_transform = self.manifold.proj(poincare_x_nei_transform, self.c)

                    attention1 = F.softmax(self.atten1(torch.cat((poincare_x_nei_transform, torch.zeros_like(poincare_x_nei_transform)), dim=-1)).squeeze(-1), dim=-1)
                    attention2 = F.softmax(self.atten2(torch.cat((poincare_x_nei_transform, torch.zeros_like(poincare_x_nei_transform)), dim=-1)).squeeze(-1), dim=-1)
                    multihead_attention = (attention1 + attention2) / 2

                    hyperboloid_x_nei_transform = self.manifold.poincare_to_hyperboloid(poincare_x_nei_transform, self.c)
                    hyperboloid_x_nei_transform = self.manifold.hyperboloid_proj(hyperboloid_x_nei_transform, self.c)
                    hyperboloid_x_final = self.manifold.hyperboloid_centroid(hyperboloid_x_nei_transform, attention)
                    hyperboloid_x_final = self.manifold.hyperboloid_proj(hyperboloid_x_final, self.c)
                    x_final = self.manifold.hyperboloid_to_poincare(hyperboloid_x_final, self.c)
                    x_final = self.manifold.proj(x_final, self.c)
                    x_final = self.act(x_final)
                    return self.manifold.proj(x_final, self.c)
                elif self.nei_agg == 2:
                    poincare_x_nei_transform = self.manifold.hyperboloid_to_poincare(hyperboloid_x_nei_transform, self.c)
                    poincare_x_nei_transform = self.manifold.proj(poincare_x_nei_transform, self.c)
                    hyperboloid_x_final = self.manifold.hyperboloid_proj(self.manifold.hyperboloid_centroid(self.manifold.hyperboloid_proj(self.manifold.poincare_to_hyperboloid(self.manifold.proj(self.MLP_f(poincare_x_nei_transform), self.c), self.c), self.c), self.c), self.c)
                    x_final = self.manifold.proj(self.MLP_fi(self.manifold.proj(self.manifold.hyperboloid_to_poincare(hyperboloid_x_final, self.c), self.c)), c=self.c)
                    return x_final
                else:
                    raise NotImplementedError("The specified neighbor aggregation type is not implemented.")
        else:
            raise NotImplementedError("The specified correlation type is not implemented.")

class KPGraphConvolution(nn.Module):
    #Hyperbolic Kernel Point Convolution Layer.
    def __init__(self, manifold, kernel_size, KP_extent, in_features, out_features, use_bias, dropout, c, nonlin, deformable, AggKlein, corr, nei_agg):
        super(KPGraphConvolution, self).__init__()
        self.net = KernelPointAggregation(kernel_size, in_features, out_features, KP_extent, manifold, use_bias, dropout, c, nonlin, aggregation_mode='sum', deformable=deformable, AggKlein=AggKlein, corr=corr, nei_agg=nei_agg)

    def forward(self, input):
        x, nei, nei_mask = input
        h = self.net(x, nei, nei_mask)
        return h, nei, nei_mask

"""
