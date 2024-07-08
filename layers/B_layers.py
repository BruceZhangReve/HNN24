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

"""
Section 2: Poincare Linears
"""

"""
#BLinear -v1

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

        # Initialize weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(torch.float64))
        self.bias = nn.Parameter(torch.Tensor(out_features).to(torch.float64)) if use_bias else None
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        # Apply dropout to weights
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        # Perform Mobius matrix-vector multiplication
        #mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        mv = self.manifold.mobius_matvec(self.weight, x, self.c)
        # Project the result
        res = self.manifold.proj(mv, self.c)

        if self.use_bias:
            # Project the bias to the tangent space
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.proj(self.manifold.expmap0(bias, self.c), self.c)
            # Add the bias
            res = self.manifold.mobius_add(res, hyp_bias, self.c)
            res = self.manifold.proj(res, self.c)

        # Apply the activation function if specified
        if self.act is not None:
            xt = self.act(self.manifold.logmap0(res, c=self.c))
            xt = self.manifold.proj_tan0(xt, c=self.c)
            res = self.manifold.proj(self.manifold.expmap0(xt, c=self.c), c=self.c)

        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}, use_bias={}, act={}'.format(
            self.in_features, self.out_features, self.c, self.use_bias, self.act)
#BLinear(manifold, in_features, out_features, c=c, dropout=0.5, act=nonlin, use_bias=True,)


"""

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
        x_tan = self.manifold.proj_tan0(x_tan, self.c)
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
"""


#BLinear -v4, 

class BLinear(nn.Module):
    def __init__(self, manifold, in_features, out_features, c, dropout, nonlin=None, use_bias=True):
        super(BLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.act = None
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)

        self.E_linear = nn.Linear(in_features, out_features, bias = False).to(torch.float64)

        self.bias = nn.Parameter(torch.Tensor(out_features).to(torch.float64)) if use_bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # Use Xavier to init E_linear's weights
        nn.init.xavier_uniform_(self.E_linear.weight, gain=nn.init.calculate_gain('relu'))
        if self.E_linear.bias is not None:
            nn.init.constant_(self.E_linear.bias, 0)

    def forward(self, x):
        # Apply dropout to the input
        x = self.dropout(x)

        #Note that expmap0/exp contains proj already!
        res = self.manifold.expmap0(
                self.E_linear(self.manifold.proj_tan0(
                                self.manifold.logmap0(x, self.c),
                                self.c)),
                self.c)

        #Note that expmap0/exp contains proj already!
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            res = self.manifold.proj(self.manifold.mobius_add(res, #then mobius addition
                                    self.manifold.expmap0(bias, self.c), self.c), # a bias on manifold
                                    self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}, use_bias={}, act={}, dropout_rate={}'.format(
            self.in_features, self.out_features, self.c, self.use_bias, self.act, self.dropout_rate)
#BLinear(manifold, in_features, out_features, c=c, dropout=0.5, act=nonlin, use_bias=True)


class BAct(Module):

    def __init__(self, manifold, c, act):
        super(BAct, self).__init__()
        self.manifold = manifold
        self.c = c
        self.act = act

    def forward(self, x):
         #Note that expmap0/exp contains proj already!
        return self.manifold.expmap0(
                    self.manifold.proj_tan0(
                        self.act(self.manifold.logmap0(x, c=self.c)), 
                        c=self.c), c=self.c)

    def extra_repr(self):
        return 'c={}, act={}'.format(
            self.c, self.act
        )


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
class KernelPointAggregation(nn.Module):
    """
    Parameters:
    - kernel_size: 
    - x_nei: The result from "gather()". (n,nei_num,d)
    - nonlin: Activation functions when doing transformation eg,F.relu
    Returns:
    - res: Tensor of kernel positions.

    Remarks:
    x_kernel,x_nei are poincare tensors; res is the Klein tensor
    """
    def __init__(self, kernel_size, in_channels, out_channels, KP_extent,
                 manifold, use_bias, dropout, c, nonlin=None,
                 aggregation_mode='sum',deformable=False,AggKlein=True,corr=0):
        super(KernelPointAggregation, self).__init__()
        # Save parameters
        self.manifold = manifold
        self.c=c #Note this c is from the encoer(BKNet), should be on cuda is specified
        self.K = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KP_extent = KP_extent #This is the radius for generating kernels
        #print(AggKlein)
        self.AggKlein = AggKlein

        self.deformable = deformable
        self.corr = corr
        #print('corr=', self.corr)

        self.linears = nn.ModuleList([BLinear(manifold,in_channels,out_channels,
                                             self.c,dropout,nonlin,use_bias) 
                                      for _ in range(self.K)])

        self.act = BAct(manifold, self.c, nonlin)

        """
        # This part is not yet implemented
        if deformable:
            pass
        """
        
        # Initialize kernel points
        self.kernel_tangents = self.init_KP()

        return

    def init_KP(self):
        # Create one kernel disposition. Choose the KP distance to center thanks to the KP extent
        kernel_tangents = load_kernels(self.manifold,
                                       radius=self.KP_extent,
                                       num_kpoints=self.K,
                                       dimension=self.in_channels,
                                       c=self.c,#Note this c is from the encoer(BKNet), should be on cuda is specified
                                       random=False)

        return nn.Parameter(kernel_tangents, requires_grad=False)
    
    def get_kernel_pos(self, x, nei, nei_mask, sample, sample_num, transp, radius=None):
        """
        This function moves the kernel to each x (node).

        Parameters:
        - x: Tensor of node features.
        - nei: Neighborhood connections.
        - nei_mask: Mask for the neighborhood.
        #- sample: Sample information.
        #- sample_num: Number of samples.
        - transp: Boolean indicating whether to transport.
        - radius: Radius for the kernel, default is None.

        Returns:
        - res: Tensor of kernel positions.

        Remark:
        The result tensor is of the following shape:
            [[[kp2_atx1)^T],
              [(kp3_atx1)^T],
              ...
              [(kpK_atx1)^T],
              [(kp1_atx1)^T]],
            
              ...Matrices...
            
             [[kp2_atxn)^T],
              [(kp3_atxn)^T],
              ...
              [(kpK_atxn)^T],
              [(kp1_atxn)^T]]
        """        
    
        n, d = x.shape
        if radius is None:
            radius = self.KP_extent

        K = self.kernel_tangents.shape[0]  # Kernel size

        if not transp:
            # If kernel points stay near the origin
            res = self.manifold.expmap0(self.kernel_tangents, c=self.c).repeat(n, 1, 1)
            # Explanation: Create n kernels to be assigned to each node by repeating the kernel tangents
        else:
            # If kernel points need to be transported to other positions
            # Repeat and reshape x to create a tensor of shape (n, K-1, d)
            """
            KEY: Repeat the third component (which is d, the columns of matrices) K-1 times
            (because the number of kernel should neglect x itself).
            view(n, K - 1, d) is our ultimate target, that we have n "positioned-kernels"
            with each having K kernel points in dimension d.
            """
            x_k = x.repeat(1, 1, K - 1).view(n, K - 1, d)  # (n, k-1, d)    
            # Parallel transport kernel tangents to each node position x
            tmp = self.manifold.ptransp0(x_k, self.kernel_tangents[1:], c=self.c)        
            # Project the transported kernel points to the manifold
            tmp = self.manifold.proj(tmp, c=self.c)  # Just for numeric reasons
            # Map the projected points back to the manifold
            tmp = self.manifold.expmap(x_k, tmp, c=self.c)
            # Concatenate the transported kernel points with the original points
            res = torch.concat((tmp, x.view(n, 1, d)), 1)  # (n, k, d)

        if self.deformable:
            # Placeholder for deformable kernel logic
            # Proposal: make radius a trainable parameter to control kernel spread
            pass

        return res
    
    def get_nei_kernel_dis(self, x_kernel, x_nei):
        #Thanks to isometry, the distance calculater in Klein or Poincare does not really matter!
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
        """
        Parameters:
        - x: The origional feature matrix
        - x_nei: The result from "gather()". (n,nei_num,d)

        Returns:
        - x0: x ominus x
        - x0_nei: x_i ominus x
    
        Remarks:
        This step transport every neighborhood to the origin for KPConv
        """
        x0_nei = self.manifold.expmap0(self.manifold.ptransp0back(x.repeat(1,1, x_nei.shape[1]).view(x_nei.shape),
                                                              self.manifold.logmap(x.repeat(1,1,x_nei.shape[1]).view(x_nei.shape),x_nei,c=self.c),
                                                              c=self.c),c=self.c)
        x0 = self.manifold.origin(x.shape[-1],c=self.c).repeat(x.shape[0], 1)
        return x0, x0_nei 
        
    def apply_kernel_transform(self, x_nei):
        """
        Parameters:
        - x_nei: A matrix that describes neighbor nodes' features (n,nei_num) 

        Returns:
        - res: A tensor of shape (n,K,num_nei,d)

        Notes:
        self.linears: A list containing K independent transformations to eapplied to every neighboring nodes
    
        Remarks:
        This step does transformation for neighboring feature points
        """

        res = []
        for k in range(self.K):
            res.append(self.linears[k](x_nei).unsqueeze(1))
        return torch.concat(res, dim = 1)
    
    def avg_kernel(self, x_nei_transform, x_nei_kernel_dis, AggKlein):
        """
        Parameters:
        - klein_x_nei_transform: A tensor that contains transformed info, in klein coords#(n,K,nei_num,d')
        - x_nei_kernel_dis: A tensor that contains kernel-feature pair wise distance, in poincare coords #(n,K,nei_num)

        Returns:
        - res: A tensor of shape (n,nei_num,d'),the result for inner aggregation!

        Remarks:
        This step is in fact the inner aggregation!
        """
        x_nei_transform = x_nei_transform.swapaxes(1, 2) # (n, nei_num, k, d')
        x_nei_kernel_dis = x_nei_kernel_dis.swapaxes(1, 2)# (n, nei_num, k)
        x_nei_kernel_dis = F.softmax(x_nei_kernel_dis, dim=-1)# (n, nei_num, k)
        if self.AggKlein == True:
            return self.manifold.klein_midpoint(x_nei_transform, x_nei_kernel_dis) #(n, nei_num, d')
        else:
            return self.manifold.hyperboloid_centroid(x_nei_transform, self.c ,x_nei_kernel_dis)

    #Note: We have not touched this part yet
    """
    def sample_nei(self, nei, nei_mask, sample_num):
        new_nei = []
        new_nei_mask = []
        for i in range(len(nei)):
            tot = nei_mask[i].sum()
            if tot > 0:
                new_nei.append(nei[i][torch.randint(0, tot, (sample_num,))])
                new_nei_mask.append(torch.ones((sample_num,), device=nei.device))
            else:
                new_nei.append(torch.zeros((sample_num,), device=nei.device))
                new_nei_mask.append(torch.zeros((sample_num,), device=nei.device))
        return torch.stack(new_nei).type(torch.long), torch.stack(new_nei_mask).type(torch.long)
    """
        
    def forward(self, x, nei, nei_mask, transp = True, sample = False, sample_num = 16):
        # x (n, d) data value, feature points on the manifold
        # nei (n, nei_num) neighbors
        # nei_mask (n, nei_num) 0/1 mask for neighbors
        if sample:
            nei, nei_mask = self.sample_nei(nei, nei_mask, sample_num)
        
        x_nei = gather(x, nei) # (n, nei_num, d)
        
        if transp:
            x, x_nei = self.transport_x(x, x_nei)
               
        n, nei_num, d = x_nei.shape

        #Careful with this one, not sure whether I'm correct
        kernels=self.get_kernel_pos(x, nei, nei_mask, sample, sample_num, transp= not transp) # (n, k, d) 
        #Confuse: if we transport_x back to the origin, then we shouldn't transport kernels to x's, original HKN code has a mistake

        if self.corr == 0:
            #Use d(xi ominus x, xk)
            x_nei_kernel_dis = self.get_nei_kernel_dis(kernels, x_nei)  # (n, k, nei_num)
            nei_mask = nei_mask.repeat(1, 1, self.K).view(n, self.K, nei_num)
            x_nei_kernel_dis = x_nei_kernel_dis * nei_mask  # (n, k, nei_num)
            x_nei_transform = self.apply_kernel_transform(x_nei) #(n,K,nei_num,d)
        else:
            #print('corr == 1') #Use d(xik, xk)
            x_nei_transform = self.apply_kernel_transform(x_nei) #(n,K,nei_num,d)
            if x_nei.shape[-1] != x_nei_transform.shape[-1]:
                raise ValueError("Don't change dimension in linear transformation step if use corr==1")
            x_nei_kernel_dis = self.get_nei_kernel_dis(kernels, x_nei_transform)  # (n, k, nei_num)
            nei_mask = nei_mask.repeat(1, 1, self.K).view(n, self.K, nei_num)
            x_nei_kernel_dis = x_nei_kernel_dis * nei_mask  # (n, k, nei_num)


        if self.AggKlein == True:
            #print("Using Klein Midpoint for Aggregation")
            klein_x_nei_transform = self.manifold.poincare_to_klein(x_nei_transform,c=self.c)#(n,K,nei_num,d')
            klein_x_nei_transform = self.manifold.klein_proj(klein_x_nei_transform, self.c)#Numerical reasons
            klein_x_nei_transform = self.avg_kernel(klein_x_nei_transform, x_nei_kernel_dis, self.AggKlein)#inner_agg#(n,nei_num,d') in Klein
            #print(klein_x_nei_transform.shape)
            klein_x_final = self.manifold.klein_midpoint(klein_x_nei_transform)#outer_agg#(n,d')# in Klein
            #klein_x_final = self.manifold.klein_proj(klein_x_final, self.c)#Add this sentence gives nan!
            x_final = self.manifold.klein_to_poincare(klein_x_final,c=self.c)#(n,d')# in Poincare
            x_final = self.manifold.proj(x_final,c=self.c) #Numerical reasons
        else:
            #print("Using Hyperboloid Centroid for Aggregation")
            hyperboloid_x_nei_transform = self.manifold.poincare_to_hyperboloid(x_nei_transform,c=self.c)#(n,K,nei_num,d')
            hyperboloid_x_nei_transform = self.manifold.hyperboloid_proj(hyperboloid_x_nei_transform, self.c) #Numerical reasons
            hyperboloid_x_nei_transform = self.avg_kernel(hyperboloid_x_nei_transform, x_nei_kernel_dis, not self.AggKlein)#inner_agg#(n,nei_num,d') on hyperboloid
            #print("After InngerAgg")
            hyperboloid_x_final = self.manifold.hyperboloid_centroid(hyperboloid_x_nei_transform,c=self.c)#outer_agg#(n,d')# on hyperboloid
            x_final = self.manifold.hyperboloid_to_poincare(hyperboloid_x_final,c=self.c)#(n,d')# in Poincare
            x_final = self.manifold.proj(x_final,c=self.c) #Numerical reasons

        x_final = self.act.forward(x_final)

        return x_final


#This is a simple pack up of KPAgg
class KPGraphConvolution(nn.Module):
    """
    Hyperbolic Kernel Point Convolution Layer.
    """
    def __init__(self, manifold, kernel_size, KP_extent, in_features, out_features, 
                 use_bias, dropout, c, nonlin, deformable, AggKlein, corr):
        super(KPGraphConvolution, self).__init__()
        #print(AggKlein)
        self.net = KernelPointAggregation(kernel_size, in_features, out_features, KP_extent, manifold, 
                                          use_bias, dropout, c, nonlin, aggregation_mode = 'sum',
                                          deformable = deformable, AggKlein = AggKlein, corr = corr)

    def forward(self, input):
        x, nei, nei_mask = input
        h = self.net(x, nei, nei_mask)
        output = h, nei, nei_mask
        return output

