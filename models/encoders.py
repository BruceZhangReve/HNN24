"""Graph encoders."""
import sys
sys.path.append('/data/lige/HKN')# Please change accordingly!

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
from layers.att_layers import GraphAttentionLayer
import layers.B_layers as B_layers
import layers.hyp_layers as hyp_layers
from layers.layers import GraphConvolution, Linear, get_dim_act
import utils.math_utils as pmath

from geoopt import ManifoldParameter


class Encoder(nn.Module):
    """
    Encoder abstract class.
    """

    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.kp:
            #print(f"adj: {adj}")
            nei, nei_mask = adj
            input = (x, nei, nei_mask)
            output, _, __ = self.layers.forward(input)
            #Actually corresponds to (h, nei, nei_mask), but we only need h as output
        elif self.encode_graph:
            #Not sure what this means
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output

class MLP(Encoder):
    """
    Multi-layer perceptron.
    """

    def __init__(self, c, args):
        super(MLP, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False
        self.kp = False

class HKPNet(Encoder):
    """
    HKPNet.
    """

    def __init__(self, c, args):
        super(HKPNet, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        if args.linear_before != None:
            self.before = True
            args.linear_before = int(args.linear_before)
            self.linear_before = hyp_layers.LorentzLinear(self.manifold, dims[0], args.linear_before, args.bias, args.dropout)
            dims[0] = args.linear_before
        else:
            self.before = False
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.KPGraphConvolution(
                            self.manifold, args.kernel_size, args.KP_extent, args.radius, in_dim, out_dim, args.bias, args.dropout, nonlin=act if i != 0 else None, deformable = args.deformable
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.kp = True

    def encode(self, x, adj):
        if self.before:
            x = self.linear_before(x)
            
        # x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        # x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        # x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HKPNet, self).encode(x, adj)

class HyboNet(Encoder):
    """
    HyboNet.
    """

    def __init__(self, c, args):
        super(HyboNet, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.LorentzGraphConvolution(
                            self.manifold, in_dim, out_dim, args.bias, args.dropout, args.use_att, args.local_agg, nonlin=act if i != 0 else None
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.kp = False

    def encode(self, x, adj):
        # x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        # x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        # x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HyboNet, self).encode(x, adj)


class HNN(Encoder):
    """
    Hyperbolic Neural Networks.
    """

    def __init__(self, c, args):
        super(HNN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, _ = hyp_layers.get_dim_act_curv(args)
        hnn_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hnn_layers.append(
                    hyp_layers.HNNLayer(
                            self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
            )
        self.layers = nn.Sequential(*hnn_layers)
        self.encode_graph = False
        self.kp = False

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return super(HNN, self).encode(x_hyp, adj)

class GCN(Encoder):
    """
    Graph Convolution Networks.
    """

    def __init__(self, c, args):
        super(GCN, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gc_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            gc_layers.append(GraphConvolution(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*gc_layers)
        self.encode_graph = True
        self.kp = False


class HGCN(Encoder):
    """
    Hyperbolic-GCN.
    """

    def __init__(self, c, args):
        super(HGCN, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att, args.local_agg
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True
        self.kp = False

    def encode(self, x, adj):
        x_tan = self.manifold.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
        return super(HGCN, self).encode(x_hyp, adj)


class GAT(Encoder):
    """
    Graph Attention Networks.
    """

    def __init__(self, c, args):
        super(GAT, self).__init__(c)
        assert args.num_layers > 0
        dims, acts = get_dim_act(args)
        gat_layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            assert dims[i + 1] % args.n_heads == 0
            out_dim = dims[i + 1] // args.n_heads
            concat = True
            gat_layers.append(
                    GraphAttentionLayer(in_dim, out_dim, args.dropout, act, args.alpha, args.n_heads, concat))
        self.layers = nn.Sequential(*gat_layers)
        self.encode_graph = True
        self.kp = False


class Shallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(Shallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.lorentz = manifolds.Lorentz()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            weights = self.manifold.init_weights(weights, self.c)
            trainable = True
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = manifolds.ManifoldParameter(weights, trainable, self.manifold, self.c)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            if self.use_feats:
                dims[0] = args.feat_dim + weights.shape[1]
            else:
                dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False
        self.kp = False

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        h = self.lorentz.logmap0(h)
        if self.use_feats:
            h = torch.cat((h, x), 1)
        return super(Shallow, self).encode(h, adj)


class LorentzShallow(Encoder):
    """
    Shallow Embedding method.
    Learns embeddings or loads pretrained embeddings and uses an MLP for classification.
    """

    def __init__(self, c, args):
        super(LorentzShallow, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.use_feats = args.use_feats
        weights = torch.Tensor(args.n_nodes, args.dim)
        if not args.pretrained_embeddings:
            trainable = True
            weights = self.manifold.random_normal(weights.shape, std=1./math.sqrt(weights.shape[-1]))
        else:
            weights = torch.Tensor(np.load(args.pretrained_embeddings))
            assert weights.shape[0] == args.n_nodes, "The embeddings you passed seem to be for another dataset."
            trainable = False
        self.lt = ManifoldParameter(weights, manifold=self.manifold, requires_grad=trainable)
        self.all_nodes = torch.LongTensor(list(range(args.n_nodes)))
        layers = []
        if args.pretrained_embeddings is not None and args.num_layers > 0:
            # MLP layers after pre-trained embeddings
            dims, acts = get_dim_act(args)
            dims[0] = weights.shape[1]
            for i in range(len(dims) - 1):
                in_dim, out_dim = dims[i], dims[i + 1]
                act = acts[i]
                layers.append(hyp_layers.LorentzLinear(self.manifold, in_dim, out_dim, args.bias, args.dropout, 10, nonlin=act if i != 0 else None))
        self.layers = nn.Sequential(*layers)
        self.encode_graph = False
        self.kp = False
        if args.use_feats:
            self.transform = hyp_layers.LorentzLinear(self.manifold, args.feat_dim + 1, args.dim, args.bias, args.dropout, 10)

    def encode(self, x, adj):
        h = self.lt[self.all_nodes, :]
        if self.use_feats:
            ones = torch.ones(x.shape[:-1] + (1, ), device=x.device)
            x = torch.cat([ones, x], dim=-1)
            x = self.manifold.expmap0(x)
            x = self.transform(x)
            h = self.manifold.projx(x + h)

        return super(LorentzShallow, self).encode(h, adj)

############################NEWLY_ADDED############################
class BMLP(Encoder):
    def __init__(self, c, args):
        super(BMLP, self).__init__(c) #Then we have self.c on cuda as initialized by Encoder class
        self.manifold = getattr(manifolds, args.manifold)()#Initialize a manifold, like a PoincareBall
        assert args.num_layers > 1
        dims, acts, self.curvatures = B_layers.get_dim_act_curv(args) #Automatically set dim and curvatures in KPconv layers
        #This curvature is already a list ocntaining num_layers tensors namely c on cuda(curvatures for encoder)
        self.curvatures.append(self.c)#Now it contains (n+1) tensors namely c on cuda
        hgc_layers = []
        for i in range(len(dims) - 1):
            #We currently not doing trainable curvatures like HGCN
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            #print(act)
            hgc_layers.append(
                #Need to ensure things works if act==None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    B_layers.BMLP(
                            self.manifold,in_dim,out_dim, self.c, args.dropout, 
                            act, args.bias
                            )
            )
        self.layers = nn.Sequential(*hgc_layers)

        self.encode_graph = True
        self.kp = False

    def encode(self, x, adj):
        x=x.to(torch.float64)##This is an ugly fix here also didn't work as well
        x_tan = self.manifold.proj_tan0(x, self.c)
        x_hyp = self.manifold.expmap0(x_tan, c=self.c)
        x_hyp = self.manifold.proj(x_hyp, c=self.c)
        return super(BMLP, self).encode(x_hyp, adj)

class BKNet(Encoder):
    """
    BKNet.
    """
    def __init__(self, c, args):
        super(BKNet, self).__init__(c) #Then we have self.c on cuda as initialized by Encoder class
        self.manifold = getattr(manifolds, args.manifold)()#Initialize a manifold, like a PoincareBall
        assert args.num_layers > 1
        dims, acts, self.curvatures = B_layers.get_dim_act_curv(args) #Automatically set dim and curvatures in KPconv layers
        #This curvature is already a list ocntaining num_layers tensors namely c on cuda(curvatures for encoder)
        #self.curvatures.append(self.c)#Now it contains (n+1) tensors namely c on cuda
        hgc_layers = []
        #Whether to add linear layers before KPConv
        if args.linear_before != None:
            self.before = True
            args.linear_before = int(args.linear_before)
            self.linear_before = B_layers.BLinear(self.manifold, dims[0], args.linear_before, self.c,
                                                     args.dropout, acts[0], args.bias)
            dims[0] = args.linear_before
        else:
            self.before = False

        #Initiate a sequence of Poincare KPConv layers
        for i in range(len(dims) - 1):
            #We currently not doing trainable curvatures like HGCN
            #c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            #print(act)
            hgc_layers.append(
                #Need to ensure things works if act==None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    B_layers.KPGraphConvolution(
                            self.manifold, args.kernel_size, args.KP_extent, in_dim, out_dim, args.bias,
                            args.dropout, self.curvatures[i], nonlin=act, deformable = args.deformable
                    )
            )
        self.layers = nn.Sequential(*hgc_layers)
        
        #stop here
        self.encode_graph = True
        self.kp = True

    def encode(self, x, adj):
        x=x.to(torch.float64)##This is an ugly fix here also didn't work as well
        x_tan = self.manifold.proj_tan0(x, self.c)
        x_hyp = self.manifold.expmap0(x_tan, c=self.c)
        x_hyp = self.manifold.proj(x_hyp, c=self.c)

        if self.before:
            x_hyp = self.linear_before(x_hyp)
        #print(f"adj: {adj}")
        return super(BKNet, self).encode(x_hyp, adj)