import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    def __init__(self, hid_dim, s_dim):
        super().__init__()

        self.w_ur1 = nn.Linear(hid_dim, hid_dim)
        self.w_ur2 = nn.Linear(hid_dim, hid_dim)
        self.w_vr1 = nn.Linear(hid_dim, hid_dim)
        self.w_vr2 = nn.Linear(hid_dim, hid_dim)
        self.w_uv1 = nn.Linear(hid_dim * 2, hid_dim)
        self.w_uv2 = nn.Linear(hid_dim, s_dim)
        self.w_uv3 = nn.Linear(s_dim, 1)

        self.bn1 = nn.BatchNorm1d(hid_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(hid_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(hid_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(s_dim, momentum=0.5)

    def forward(self, embeds_u, embeds_v, out_put=True):
        x_u = F.relu((self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)

        x_v = F.relu(self.w_vr1(embeds_v))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.w_uv1(x_uv))
        x = F.dropout(x, training=self.training)

        x = F.relu(self.w_uv2(x))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)

        if not out_put:
            return x
        else:
            return scores.squeeze()

class Attention(torch.nn.Module):
    def __init__(self, ModelSettings):
        super(Attention, self).__init__()
        self.sim_func = ModelSettings['sim_func']
        hid_dim = eval(ModelSettings['att_input_dim'])
        if self.sim_func == 'dot':
            pass
        elif self.sim_func == 'concat':
            self.w = nn.Linear(hid_dim * 2, hid_dim)
            self.v = nn.Linear(hid_dim, 1)
        elif self.sim_func == 'add':
            self.w = nn.Linear(hid_dim, hid_dim)
            self.u = nn.Linear(hid_dim, hid_dim)
            self.v = nn.Linear(hid_dim, 1)
        elif self.sim_func == 'linear':
            self.w = nn.Linear(hid_dim, hid_dim)
        self._init_weights()

    def forward(self, query, seq, seq_lens=None):
        if self.sim_func == 'dot':
            query = query.squeeze().unsqueeze(1)
            a = self.mask_softmax(torch.bmm(seq, query.permute(0, 2, 1)), seq_lens, 1)
            return a
        elif self.sim_func == 'concat': 
            seq_len = len(seq[0])
            batch_size = len(seq)
            query = query.squeeze().unsqueeze(1)
            a = torch.cat([seq, query.repeat([1, seq_len, 1])], 2).reshape([seq_len * batch_size, -1])
            a = F.relu(self.w(a))
            a = F.relu(self.v(a))
            a = self.mask_softmax(a.reshape([batch_size, seq_len, 1]), seq_lens, 1)
            return a
        elif self.sim_func == 'add':
            seq_len = len(seq[0])
            batch_size = len(seq)
            seq = self.w(seq.reshape([batch_size * seq_len, -1]))
            query = self.u(query).repeat([seq_len, 1])
            a = self.mask_softmax(self.v(F.tanh(seq + query)).reshape([batch_size, seq_len, 1]), seq_lens, 1)
            return a
        elif self.sim_func == 'linear':
            seq_len = len(seq[0])
            batch_size = len(seq)
            query = query.squeeze()
            query = self.w(query).unsqueeze(2)
            a = self.mask_softmax(torch.bmm(seq, query), seq_lens, 1)
            return a

    def _init_weights(self):
        if self.sim_func == 'dot':
            pass
        elif self.sim_func == 'concat':
            nn.init.uniform_(self.w.weight, a=-0.1, b=0.1)
            nn.init.uniform_(self.v.weight, a=-0.1, b=0.1)
        elif self.sim_func == 'add':
            nn.init.uniform_(self.w.weight, a=-0.1, b=0.1)
            nn.init.uniform_(self.u.weight, a=-0.1, b=0.1)
            nn.init.uniform_(self.v.weight, a=-0.1, b=0.1)
        elif self.sim_func == 'linear':
            nn.init.uniform_(self.w.weight, a=-0.1, b=0.1)
    
    def mask_softmax(self, seqs, seq_lens=None, dim=1):
        if seq_lens is None:
            res = F.softmax(seqs, dim=dim)
        else:
            max_len = len(seqs[0])
            batch_size = len(seqs)
            ones = seq_lens.new_ones(batch_size, max_len, device=seq_lens.device)
            range_tensor = ones.cumsum(dim=1)
            mask = (seq_lens.unsqueeze(1) >= range_tensor).long()
            mask = mask.float()
            mask = mask.unsqueeze(2)
            # masked_vector = seqs.masked_fill((1 - mask).byte(), -1e32)
            masked_vector = seqs.masked_fill((1 - mask).bool(), -1e32)
            res = F.softmax(masked_vector, dim=dim)
        return res
