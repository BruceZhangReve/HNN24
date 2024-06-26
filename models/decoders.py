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


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True


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
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LorentzDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.use_bias = args.bias
        self.cls = geoopt_ManifoldParameter(self.manifold.random_normal((args.n_classes, args.dim), std=1./math.sqrt(args.dim)), manifold=self.manifold)
        if args.bias:
            self.bias = nn.Parameter(torch.zeros(args.n_classes))
        self.decode_adj = False

    def decode(self, x, adj):
        return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias

class LorentzDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LorentzDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.use_bias = args.bias
        self.cls = geoopt_ManifoldParameter(self.manifold.random_normal((args.n_classes, args.dim), std=1./math.sqrt(args.dim)), manifold=self.manifold)
        if args.bias:
            self.bias = nn.Parameter(torch.zeros(args.n_classes))
        self.decode_adj = False

    def decode(self, x, adj):
        return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias

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
        self.cls = geoopt_ManifoldParameter(self.manifold.random_normal((args.n_classes, args.dim), std=1./math.sqrt(args.dim)), manifold=self.manifold)
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
        #return self.manifold.norm()


#Decoders for node classification
model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
    'HyboNet': LorentzDecoder,
    'HKPNet': LorentzDecoder,
    'LorentzShallow': LorentzDecoder,
    'BMLP': PoincareDecoder,
    'BKNet': PoincareDecoder,
}

#Decoders for link prediction
gcdecoder = {
    'HGCN': LorentzPoolDecoder,
    'HyboNet': LorentzPoolDecoder,
    'HKPNet': LorentzPoolDecoder,
    'LorentzShallow': LorentzPoolDecoder
}