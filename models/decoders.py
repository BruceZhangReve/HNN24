"""Graph decoders."""
import sys
sys.path.append('/data/lige/HKN')# Please change accordingly!

import math
import manifolds
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear
from kernels.poincare_kernel_points import simple_PoincareWrappedNormal
from kernels.poincare_kernel_points import load_kernels

from layers.B_layers import BLinear
from layers.hyp_layers import LorentzLinear as LLinear

from geoopt import ManifoldParameter as geoopt_ManifoldParameter
from manifolds.base import ManifoldParameter as base_ManifoldParameter


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs

"""
#I think there's no need to have a "decoder" for Euclidean method
class GCNDecoder(Decoder):

    #Graph Convolution Decoder.

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True
"""


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, 0, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )

class LorentzDecoder(Decoder):
    """
    MLP Decoder for HKPNet node classification models.
    """
    def __init__(self, c, args):
        super(LorentzDecoder, self).__init__(c)
        self.use_HCDist = True
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.use_bias = args.bias
        if self.use_HCDist:
            self.cls = geoopt_ManifoldParameter(self.manifold.random_normal((args.n_classes, args.dim), std=1./math.sqrt(args.dim)), manifold=self.manifold)
            if args.bias:
                self.bias = nn.Parameter(torch.zeros(args.n_classes))
        else:
            self.cls = LLinear(self.manifold, self.input_dim, self.output_dim, self.use_bias, args.dropout, nonlin=None)
        self.decode_adj = False

    def decode(self, x, adj):
        if self.use_HCDist:
            return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias
        else:
            return self.manifold.logmap0(self.cls(x))

class LorentzPoolDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean graph classification models.
    """

    def __init__(self, c, args):
        super(LorentzPoolDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.use_bias = args.bias
        if args.model == "HKPNet":
            self.cls = geoopt_ManifoldParameter(self.manifold.random_normal((args.n_classes, args.dim), std=1./math.sqrt(args.dim)), manifold=self.manifold)
        else:
            self.cls = geoopt_ManifoldParameter(self.manifold.random_normal((args.n_classes, args.dim), std=1./math.sqrt(args.dim), c = self.c), manifold=self.manifold)

        if args.bias:
            self.bias = nn.Parameter(torch.zeros(args.n_classes))
        self.decode_adj = False

    def decode(self, x, ed_idx):
        x0 = []
        for i in range(len(ed_idx)):
            x0.append(self.manifold.mid_point(x[0 if i == 0 else ed_idx[i - 1]:ed_idx[i]]))
        x0 = torch.stack(x0)
        return (2 + 2 * self.manifold.cinner(x0, self.cls)) + self.bias

############################NEWLY_ADDED############################
class EuclideanPoolDecoder(Decoder):
    def __init__(self, c, args):
        super(EuclideanPoolDecoder, self).__init__(c)
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, None, args.bias)
        self.decode_adj = True
    
    def decode(self, x, ed_idx, adj):
        input = (x, adj)
        x, _ = self.cls.forward(input)

        x0 = []
        for i in range(len(ed_idx)):
            #x0.append(x[0 if i == 0 else ed_idx[i - 1]:ed_idx[i]].mean(dim=0))
            x0.append(x[0 if i == 0 else ed_idx[i - 1]:ed_idx[i]].sum(dim=0))
        x0 = torch.stack(x0)

        return x0

class PoincareDecoder(Decoder):
    """
    MLP Decoder for Poincare node classification models.
    """
    def __init__(self, c, args):
        super(PoincareDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.use_bias = args.bias
        self.c=c
        self.origin = self.manifold.origin(args.dim, c=self.c)
        #print(f"self.origin device: {self.origin.device}") #It's on cpu yet

        self.cls = base_ManifoldParameter(self.manifold.random_normal((args.n_classes, args.dim),c=self.c, std=1./math.sqrt(args.dim)),
                                          requires_grad=True, manifold=self.manifold, c=self.c)
        #print(f"self.cls device: {self.cls.device}")#Let's check about it

        if args.bias:
            self.bias = nn.Parameter(torch.zeros(args.n_classes))
        else:
            self.bias = nn.Parameter(torch.zeros(args.n_classes),requires_grad=False)
        self.decode_adj = False

    def decode(self, x, adj):
        return (self.manifold.HCDist(x,self.cls,c=self.c)) + self.bias

"""
# PoincarePoolDecoder -trainable points
class PoincarePoolDecoder(Decoder):
    def __init__(self, c, args):
        super(PoincarePoolDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.use_bias = args.bias
        self.c = c
        self.cls_num = args.n_classes
        self.cls = base_ManifoldParameter(self.manifold.random_normal((args.n_classes, args.dim),c=self.c, std=1./math.sqrt(args.dim)),
                                          requires_grad=True, manifold=self.manifold, c=self.c)

        if args.bias:
            self.bias = nn.Parameter(torch.zeros(args.n_classes))

        self.decode_adj = False

    def decode(self, x, ed_idx):
        #Note x here is a bit different, although it is (n,d''), this n is the the concatenation of many graphs
        x0 = []
        for i in range(len(ed_idx)):
            #print(x[0 if i == 0 else ed_idx[i - 1]:ed_idx[i]])
            #It means if i==0, then use x[0:ed_idx[0]], if i==1 then use x[ed_idx[0]:ed_idx[1]] ......
            
            #Klein Aggregation you can consider as a hyperbolic global mean pool
            x0.append(self.manifold.proj(self.manifold.klein_to_poincare(self.manifold.klein_midpoint(
                                                                        self.manifold.klein_proj(self.manifold.poincare_to_klein(
                                                                        x[0 if i == 0 else ed_idx[i - 1]:ed_idx[i]],self.c),self.c)),
                                                                        self.c),
                                                                        self.c))
            
        x0 = torch.stack(x0)

        return (self.manifold.HCDist(x0, self.cls, self.c)) + self.bias #(batch_size,num_classes)
"""

"""
# PoincarePoolDecoder -fixed kernel representation
class PoincarePoolDecoder(Decoder):
    def __init__(self, c, args):
        super(PoincarePoolDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.use_bias = args.bias
        self.KP_extent = args.KP_extent
        self.c = c

        self.cls = self.init_KP()
        #print(self.cls.shape)

        if args.bias:
            self.bias = nn.Parameter(torch.zeros(args.n_classes))

        self.decode_adj = False
    

    def init_KP(self):
        return nn.Parameter(load_kernels(self.manifold, radius=self.KP_extent, num_kpoints=self.output_dim, dimension=self.input_dim, c=self.c, random=False), requires_grad=False)

    def kernel_graph_embedding(self, kernel_points, graph_points):
        ######
        #kernel_points: (num_cls, embedding_dim)
        #graph_points: (n, embedding dim)

        #return tensor: (num_cls), vector for graph
        ######
        
        kernel_points = kernel_points.unsqueeze(1).expand(-1, graph_points.size(0), -1) #(num_cls, n, embedding_dim)
        graph_points = graph_points.unsqueeze(0) #(1, n, embedding_dim)
        embedding = torch.sqrt(self.manifold.sqdist(graph_points, kernel_points, self.c)) #(num_cls, n)

        return embedding.sum(dim=-1) #(num_cls)

    def decode(self, x, ed_idx):
        #Note x here is a bit different, although it is (n,d''), this n is the the concatenation of many graphs
        x0 = []
        for i in range(len(ed_idx)):
            #print(x[0 if i == 0 else ed_idx[i - 1]:ed_idx[i]])
            #It means if i==0, then use x[0:ed_idx[0]], if i==1 then use x[ed_idx[0]:ed_idx[1]] ......
            
            x0.append(self.kernel_graph_embedding(self.cls, x[0 if i == 0 else ed_idx[i - 1]:ed_idx[i]]))
            
        x0 = torch.stack(x0)

        return x0  #(batch_size,num_classes)
"""


# PoincarePoolDecoder -tangent space perspective
class PoincarePoolDecoder(Decoder):
    def __init__(self, c, args):
        super(PoincarePoolDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.use_bias = args.bias
        self.c = c
        self.cls_num = args.n_classes
        self.linear = BLinear(self.manifold,self.input_dim,self.output_dim,
                                self.c,dropout=0.05,nonlin=None,use_bias=True) 

        self.decode_adj = False

    def decode(self, x, ed_idx):
        #Note x here is a bit different, although it is (n,d''), this n is the the concatenation of many graphs
        #First, adjust the dimension to the output dimension, without activation here!!!
        #x = self.linear(x)
        x0 = []
        for i in range(len(ed_idx)):
            #print(x[0 if i == 0 else ed_idx[i - 1]:ed_idx[i]])
            #It means if i==0, then use x[0:ed_idx[0]], if i==1 then use x[ed_idx[0]:ed_idx[1]] ......

            #Project to tangent space and use an Euclidean Average as the global mean pool
            #x0.append((self.manifold.proj_tan0(self.manifold.logmap0(x[0 if i == 0 else ed_idx[i - 1]:ed_idx[i]],self.c),self.c)).sum(dim=0))
            x0.append(self.manifold.proj(self.manifold.klein_to_poincare(self.manifold.klein_midpoint(
                                                                        self.manifold.klein_proj(self.manifold.poincare_to_klein(
                                                                        x[0 if i == 0 else ed_idx[i - 1]:ed_idx[i]],self.c),self.c)),
                                                                        self.c),
                                                                        self.c))
            
            
        #x0 = torch.stack(x0) #(batch_size,num_classes), already an Euclidean vector
        x0 = torch.stack(x0) #(batch_size,embedding dimension) yet an hyperbolic vector
        x0 = self.manifold.proj_tan0(self.manifold.logmap0(self.linear(x0),self.c),self.c)

        return x0


#Decoders for node classification
model2decoder = {
    #'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
    'HyboNet': LorentzDecoder,
    'HKPNet': LorentzDecoder,
    'LorentzShallow': LorentzDecoder,
    'BMLP': PoincareDecoder,
    'BKNet': PoincareDecoder
}

#Decoders for graph classification
gcdecoder = {
    'GCN': EuclideanPoolDecoder,
    'HGCN': LorentzPoolDecoder,
    'HyboNet': LorentzPoolDecoder,
    'HKPNet': LorentzPoolDecoder,
    'LorentzShallow': LorentzPoolDecoder,
    'BKNet': PoincarePoolDecoder,
}