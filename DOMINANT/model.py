"""
Python Source code for "Deep Anomaly Detection on Attributed Networks" (Dominant)
Paper Source: https://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf
Author: Rachel Tjarksen
This is the main model code for the paper
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution

class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x

class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GraphConvolution(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x


# dominant class
class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(Dominant, self).__init__()
        
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)
    
    # forward pass function
    def forward(self, x, adj):
        x = self.shared_encoder(x, adj)
        x_hat = self.attr_decoder(x, adj)
        struct_reconstructed = self.struct_decoder(x, adj)
        return struct_reconstructed, x_hat