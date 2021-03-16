import torch
import torch.nn as nn

import dgl

class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, aggregator_type='graphsage'):
        super(Aggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if aggregator_type == 'gnn':
            self.W_1 = nn.Linear(self.in_dim, self.out_dim)     
            self.W_2 = nn.Linear(self.in_dim, self.out_dim)     
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()


    def forward(self, mode, g, entity_embed):
        g = g.local_var()
        g.ndata['node'] = entity_embed
        g.update_all(dgl.function.copy_u('node', 'side'), dgl.function.sum('side', 'H'))

        if self.aggregator_type == 'gnn':
            neighbors = self.activation(self.W_1(g.ndata['H']))
            interactions = torch.mul(g.ndata['node'], g.ndata['H'])
            interactions = self.activation(self.W_2(interactions))
            out = neighbors + interactions
        else:
            raise NotImplementedError

        out = self.message_dropout(out)
        return out