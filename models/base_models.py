"""Base model class."""
import sys
sys.path.append('/data/lige/HKN')# Please change accordingly!

# from os import POSIX_FADV_NOREUSE
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import layers.B_layers as B_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder, gcdecoder
from utils.eval_utils import acc_f1, MarginLoss


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold

        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device) #Base model carries the curvature on cuda
        else:
            self.c = nn.Parameter(torch.Tensor([1.])) #Without specific indication, it's on cpu

        self.manifold = getattr(manifolds, self.manifold_name)()#Initialize a manifold, eg PoincareBall

        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            args.feat_dim = args.feat_dim + 1

        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)#Initialize an encoder, eg BKNet

    def encode(self, x, adj):
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)#Using geoopt
        elif self.manifold.name in ['PoincareBall']:
            x = self.manifold.expmap0(x,c=self.c)#Using manifold.base
        h = self.encoder.encode(x, adj)
        #Note: h is the updated feature points matrix of shape (n,d')
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)#Initialize the BaseModel #This is inheritence!
        self.decoder = model2decoder[args.model](self.c, args)# This is composition!
        self.margin = args.margin
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return output[idx]

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        # print(data['labels'][idx].shape, data['labels'].shape)
        if self.manifold_name == 'Lorentz':
            #Margin Loss
            #correct = output.gather(1, data['labels'][idx].to(torch.long).unsqueeze(-1))
            #loss = F.relu(self.margin - correct + output).mean()
            #CE Loss
            loss = F.cross_entropy(output, data['labels'][idx].to(torch.long), self.weights.to(output.dtype))
        elif self.manifold_name == 'PoincareBall':
            #Margin Loss
            #correct = output.gather(1, data['labels'][idx].to(torch.long).unsqueeze(-1))
            #loss = F.relu(self.margin - correct + output).mean()
            #CE Loss
            loss = F.cross_entropy(output, data['labels'][idx].to(torch.long), self.weights.to(output.dtype))
        else:
            loss = F.cross_entropy(output, data['labels'][idx], self.weights)
            #loss = F.cross_entropy(output, data['labels'][idx].to(torch.long), self.weights.to(torch.float64))
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics
    
    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]

##################NOT_YET_MODIFIED

class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.loss = MarginLoss(args.margin)

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        return -sqdist

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        preds = torch.stack([pos_scores, neg_scores], dim=-1)
        loss = self.loss(preds)

        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])
##################NOT_YET_MODIFIED

class GCModel(BaseModel):
    """
    Base model for graph classification task.
    """

    def __init__(self, args):
        super(GCModel, self).__init__(args)
        self.manifold = getattr(manifolds, args.manifold)()
        #Load corresponding decoder if it is an encoder-decoder model
        try:
            self.decoder = gcdecoder[args.model](self.c, args)
        except KeyError:
            self.decoder = None

        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, ed_idx, adj=None):
        if self.manifold_name == 'Lorentz' or self.manifold_name == 'Hyperboloid':
            output = self.decoder.decode(h, ed_idx)
            #print('partially decoded value shape:',output.shape) #(min{batch_size,remaining samples},num_classes)
            return F.log_softmax(output.unsqueeze(0), dim = 1)
        elif self.manifold_name == 'PoincareBall':
            #output = self.decoder.decode(h, ed_idx) #output:(min{batch_size,remaining samples},num_classes) like the one hot encoding
            #print('partially decoded value shape:',output.shape) #
            output = self.decoder.decode(h, ed_idx)
            return  F.log_softmax(output.unsqueeze(0), dim = 1)
        elif self.manifold_name == 'Euclidean':
            return  self.decoder.decode(h, ed_idx, adj)
        else:
            raise NotImplementedError("manifold not supported")

    def compute_metrics(self, embeddings, labels, ed_idx, type = 1, adj=None):
        if self.manifold_name == 'Lorentz' or self.manifold_name == 'Hyperboloid':
            output = self.decode(embeddings, ed_idx).squeeze()
            #print(output.shape) #(batch_size, num_cls)
            #print(torch.tensor(labels)) #(n)
            loss = F.nll_loss(output, labels, self.weights)
            #loss = F.cross_entropy(output, labels, self.weights)
            acc, f1 = acc_f1(output, labels, average=self.f1_average)
            if type == 1:
                metrics = {'loss': loss, 'acc': acc, 'f1': f1}
            elif type == 2:
                metrics = {'loss': loss, 'output': output.detach()}
            return metrics

        elif self.manifold_name == 'PoincareBall':
            #print('coming here')
            output = self.decode(embeddings, ed_idx).squeeze()
            #print(output.shape,labels.shape) #(batch_size,num_class),(batch_size)
            loss = F.nll_loss(output, labels, self.weights)
            #loss = F.cross_entropy(output, labels, self.weights)
            acc, f1 = acc_f1(output, labels, average=self.f1_average)
            if type == 1:
                metrics = {'loss': loss, 'acc': acc, 'f1': f1}
            elif type == 2:
                metrics = {'loss': loss, 'output': output.detach()}
            return metrics

        elif self.manifold_name == 'Euclidean':
            output = self.decode(embeddings, ed_idx, adj)
            #print(output.shape,labels.shape)
            loss = F.cross_entropy(output, labels, self.weights)
            acc, f1 = acc_f1(output, labels, average=self.f1_average)
            if type == 1:
                metrics = {'loss': loss, 'acc': acc, 'f1': f1}
            elif type == 2:
                metrics = {'loss': loss, 'output': output.detach()}
            return metrics
        else:
             raise NotImplementedError("manifold not supported")
             return 0

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


        