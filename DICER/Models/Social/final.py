import numpy as np, pandas as pd
import tqdm, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from Models.utils.layer import Predictor, Attention
from Models.engine import Engine
from Models.Graph.utils import Aggregator



class FinalNet(nn.Module):
    def __init__(self, Sampler, ModelSettings):
        super().__init__()
        
        # init args 
        self.num_candidate, self.num_user = Sampler.candidate_num, Sampler.user_num
        self.user_neigh = Sampler.user_neigh_dict
        self.item_neigh = Sampler.item_neigh_dict
        self.user_friends = Sampler.user_friend_dict
        self.hid_dim = eval(ModelSettings['hidden_dim'])
        embed_dim = eval(ModelSettings['embed_dim'])
        dropout = eval(ModelSettings['dropout'])
        self.num_layer=eval(ModelSettings['num_layer'])
        attn_drop=eval(ModelSettings['attn_drop'])
        aggregator_type = ModelSettings['aggregator_type']
        self.fusion_type = ModelSettings['fusion_type']
        self.f_fusion_type =ModelSettings['f_fusion_type']

        self.activation = nn.LeakyReLU()

        # init embeddings layer
        self.user_embedding = nn.Embedding( self.num_user+1, embed_dim)
        self.item_embedding = nn.Embedding( self.num_candidate+1, embed_dim )

        self.user_soc_GNNs = nn.ModuleList()
        for i in range(self.num_layer):
            self.user_soc_GNNs.append(Aggregator(embed_dim, embed_dim, attn_drop, aggregator_type=aggregator_type))
        
        self.user_sim_GNNs = nn.ModuleList()
        for i in range(self.num_layer):
            self.user_sim_GNNs.append(Aggregator(embed_dim, embed_dim, attn_drop, aggregator_type=aggregator_type))

        self.item_sim_GNNs = nn.ModuleList()
        for i in range(self.num_layer):
            self.item_sim_GNNs.append(Aggregator(embed_dim, embed_dim, attn_drop, aggregator_type=aggregator_type))

        self.user_friend_att = Attention(ModelSettings)

        all_layer = self.num_layer + 1
        s_dim = 48
        self.all_layer = all_layer
        self.Predictor_1 = Predictor(self.hid_dim*all_layer, s_dim)
        self.Predictor_2 = Predictor(self.hid_dim*all_layer, s_dim)
        self.Predictor_3 = Predictor(self.hid_dim*all_layer, s_dim)

        self.init_weights()

    def graph_aggregate(self, g, GNNs, node_embedding, mode='train', Type=''):
        g = g.local_var()
        init_embed = node_embedding

        # run GNN
        all_embed = [init_embed]
        for l in range(self.num_layer):
            GNN_layer = GNNs[l]
            init_embed = GNN_layer(mode, g, init_embed)
            norm_embed = F.normalize(init_embed, p=2, dim=1)
            all_embed.append(norm_embed)

        all_embed = torch.cat(all_embed, dim=1)
        return all_embed

    def get_item_history(self, target_user_embed, node_embeddings):
        batch_size = len(target_user_embed)

        ## fusion
        item_users_pad = torch.LongTensor(self.item_users).to(target_user_embed.device)
        item_users_embed = nn.functional.embedding(item_users_pad, node_embeddings) # [B, R, H]

        if self.fusion_type == 'max':
            target_user_embed = target_user_embed.unsqueeze(1)
            target_user_embed = target_user_embed.repeat(1, item_users_embed.shape[1], 1) #[B, R, H]
            item_fusion_embeddings = torch.max(item_users_embed * target_user_embed, dim=1)[0] #[B, H]
        elif self.fusion_type == 'attention':
            a = self.item_history_att(target_user_embed, item_users_embed, torch.LongTensor(self.item_users_lens).to(target_user_embed.device)) #[B, R]
            a = torch.reshape(a, [batch_size, 1, -1])  #[B, 1, R]
            item_fusion_embeddings = torch.bmm(a, item_users_embed).squeeze() # [B, 1, R] * [B, R, H] = [B, H]
        else:
            raise ValueError('unknow fusion type: ' + self.fusion_type)
        return item_fusion_embeddings

    def get_user_history(self, target_item_embed, node_embeddings): 
        batch_size = len(target_item_embed)

        user_items_pad = torch.LongTensor(self.user_items).to(target_item_embed.device)
        user_items_embed = nn.functional.embedding(user_items_pad, node_embeddings) # [B, R, H]

        if self.fusion_type == 'max':
            target_item_embed = target_item_embed.unsqueeze(1)
            target_item_embed = target_item_embed.repeat(1, user_items_embed.shape[1], 1) #[B, R, H]
            user_fusion_embeddings = torch.max(user_items_embed * target_item_embed, dim=1)[0]
        elif self.fusion_type == 'attention':
            a = self.user_history_att(target_item_embed, user_items_embed, torch.LongTensor(self.user_items_lens).to(target_item_embed.device)) #[B, R]
            a = torch.reshape(a, [batch_size, 1, -1])  #[B, 1, R]
            user_fusion_embeddings = torch.bmm(a, user_items_embed).squeeze() # [B, 1, R] * [B, R, H] = [B, H]
        else:
            raise ValueError('unknow fusion type: ' + self.fusion_type)
        return user_fusion_embeddings
   
    def get_friend_fusion(self, active_user_embed, target_items_embed, node_embeddings):
        batch_size = active_user_embed.shape[0]
        
        f_items = torch.LongTensor(self.friends_items).to(active_user_embed.device)    # [B, F, R]
        f_items_embed = nn.functional.embedding(f_items, node_embeddings)   # [B, F, R, D]
        target_items_embed = target_items_embed.view(batch_size, 1, 1, -1)                # [B, 1, 1, D]
        f_items_embed = torch.max(f_items_embed * target_items_embed.repeat(1, f_items.shape[1], f_items.shape[2], 1) , dim=2)[0]    # [B, F, D]

        friends_lens = deepcopy(self.friends_lens)

        if self.f_fusion_type == 'attention':
            a = self.user_friend_att(active_user_embed, f_items_embed, torch.LongTensor(friends_lens).to(target_items_embed.device)) #[B, F]
            a = torch.reshape(a, [batch_size, 1, -1])  #[B, 1, F]
            f_items_embed = torch.bmm(a, f_items_embed).squeeze() # [B, 1, F] * [B, F, H] = [B, H]
        else:
            raise ValueError('unknow fusion type: ' + self.fusion_type)

        return f_items_embed

    def forward(self, user, candidate, user_soc_g, user_sim_g, item_sim_g, mode):
        """
        user: (batch_size)
        candidate: (batch_size, candidate_num)
        """
        user = user.squeeze()
        candidate = candidate.squeeze()
        batch_size = len(user)
        
        ### embedding layer
        user_id_embedding = self.user_embedding(user_soc_g.ndata['id'])
        item_id_embedding = self.item_embedding(item_sim_g.ndata['id'])

        ### user social
        user_soc_embedding = self.graph_aggregate(user_soc_g, self.user_soc_GNNs, user_id_embedding, Type='user_soc')
        user_soc_embedding[0] = torch.zeros_like(user_soc_embedding[0])
        # user sim
        user_sim_embedding = self.graph_aggregate(user_sim_g, self.user_sim_GNNs, user_id_embedding, Type='user_sim')
        user_sim_embedding[0] = torch.zeros_like(user_sim_embedding[0])
        # item sim
        item_sim_embedding = self.graph_aggregate(item_sim_g, self.item_sim_GNNs, item_id_embedding, Type='item_sim')
        item_sim_embedding[0] = torch.zeros_like(item_sim_embedding[0])
        
        ### Global info
        user_global_embedding = 0.5 * (user_sim_embedding + user_soc_embedding)
        user_global = user_global_embedding[user]
        item_sim_global = item_sim_embedding[candidate]

        ### local info
        item_local = self.get_item_history(user_global, user_global_embedding)
        user_sim_local = self.get_user_history(item_sim_global, item_sim_embedding)
        user_fri_local = self.get_friend_fusion(user_sim_local, item_sim_global, item_sim_embedding)
        user_local = 0.5*(user_sim_local + user_fri_local)

        scores_list = [
            ## global match global
            self.Predictor_1(user_global, item_sim_global),

            ## local match global
            self.Predictor_2(user_local, item_sim_global),
            self.Predictor_3(user_global, item_local),
        ]

        if mode == 'train':
            return scores_list
        else:
            scores_list = [x.view(-1,1) for x in scores_list]
            return torch.mean(torch.cat(scores_list, dim=1), dim=1)


    def init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)


class FinalEngine(Engine):
    
    def __init__(self, Sampler, DataSettings, TrainSettings, ModelSettings, ResultSettings):
        self.Sampler = Sampler
        self.model = FinalNet(Sampler, ModelSettings)
        self.model.to(TrainSettings['device'])
        self.batch_size = eval(TrainSettings['batch_size'])
        self.eval_neg_num = eval(DataSettings['eval_neg_num'])
        self.eval_ks = eval(TrainSettings['eval_ks'])
        self.user_history_sample_num = eval(DataSettings['user_history_sample_num'])
        self.item_history_sample_num = eval(DataSettings['item_history_sample_num'])
        self.user_friend_sample_num = eval(DataSettings['user_friend_sample_num'])
        super(FinalEngine, self).__init__(TrainSettings, ModelSettings, ResultSettings)

        ### neighbor data 
        self.user_neigh = Sampler.user_neigh_dict
        self.item_neigh = Sampler.item_neigh_dict
        self.user_friends = Sampler.user_friend_dict

    def history_sample(self, ks, neigh_dict, masks, mode='train', Type='user'):
        k_neighs = [ [] if x not in neigh_dict.keys() else deepcopy(neigh_dict[x]) for x in ks ]
        if Type == 'user':
            h_sample_num = self.user_history_sample_num
        else:
            h_sample_num = self.item_history_sample_num

        neigh_lens = []
        r_max = 0
        for i in range(len(k_neighs)):
            mask = masks[i]
            if mask in k_neighs[i]: 
                k_neighs[i].remove(mask) 

            if k_neighs[i] == []:
                k_neighs[i] = [0]
            r_max = max(r_max, len(k_neighs[i]))
            neigh_lens.append(len(k_neighs[i]))

            # if mode == 'train': 
            #     sample_num = min(h_sample_num, len(k_neighs[i]))
            #     k_neighs[i] = random.sample(k_neighs[i], sample_num)

        ### padding
        for i in range(len(k_neighs)):
            cur_len = len(k_neighs[i])
            k_neighs[i].extend( [0]*(r_max-cur_len) )

        return k_neighs, neigh_lens

    def friend_sample(self, users, user_social, user_history, is_sample=True):
        f_sample_num = self.user_friend_sample_num
        h_sample_num = self.user_history_sample_num
        batch_size = len(users)
        u_friends = [ [] if x not in user_social.keys() else deepcopy(user_social[x]) for x in users ]
        friends_items = []
        f_max, r_max = 0, 0
        friends_lens = []
        for i in range(batch_size):
            cur_f_items = []

            cur_friend = u_friends[i]
            if is_sample:
                sample_num = min(f_sample_num, len(cur_friend))
                cur_friend = random.sample(cur_friend, sample_num)
            f_max = max(len(cur_friend), f_max)

            for f in cur_friend:
                if f in user_history.keys():

                    tmp_items = user_history[f]
                    if is_sample:
                        sample_num = min(h_sample_num, len(tmp_items))
                        tmp_items = random.sample(tmp_items, sample_num)
                    r_max = max(len(tmp_items), r_max)
                    cur_f_items.append(tmp_items)
            
            if cur_f_items == []:
                cur_f_items = [[0]]
            friends_items.append(cur_f_items)
            friends_lens.append(len(cur_friend))
        
        ## padding
        for i in range(len(friends_items)):
            cur_f_len = len(friends_items[i])
            for j in range(cur_f_len):
                friends_items[i][j].extend( [0]*(r_max-len(friends_items[i][j])) )
            friends_items[i].extend([[0]*r_max]*(f_max-j-1))
        
        return friends_items, friends_lens
                 
    def train(self, train_loader, graphs, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        device = self.device
        user_soc_g = graphs['user_soc_g'].to(torch.device(device))
        user_sim_g = graphs['user_sim_g'].to(torch.device(device))
        item_sim_g = graphs['item_sim_g'].to(torch.device(device))

        total_loss = 0
        tmp_train_loss = []
        t0 = time.time()
        for i, input_list in enumerate(tqdm.tqdm(train_loader, desc="train", smoothing=0, mininterval=1.0)):
            batch_users, batch_items = list(input_list[1].numpy()), list(input_list[2].numpy())
            self.model.item_users, self.model.item_users_lens = self.history_sample(batch_items, self.item_neigh, batch_users, mode='train', Type='item')
            self.model.user_items, self.model.user_items_lens = self.history_sample(batch_users, self.user_neigh, batch_items, mode='train', Type='user')
            self.model.friends_items, self.model.friends_lens = self.friend_sample(batch_users, self.user_friends, self.user_neigh)

            # run model
            input_list = [x.to(device) for x in input_list]
            self.optimizer.zero_grad()
            pred_list= self.model(*input_list[1:], user_soc_g, user_sim_g, item_sim_g, mode='train')

            # loss
            with torch.autograd.set_detect_anomaly(True):
                label = input_list[0]
                loss = 0
                for pred in pred_list:
                    loss += self.criterion(pred.squeeze(), label.float())
                loss.backward(retain_graph=False)
                self.optimizer.step()
            tmp_train_loss.append(loss.item())
            total_loss += loss.item()

        t1 = time.time()
        print("Epoch ", epoch_id, " Train cost:", t1-t0, " Loss: ", np.mean(tmp_train_loss))
        return np.mean(tmp_train_loss) 

    def evaluate(self, eval_pos_loader, eval_neg_loader, graphs, epoch_id, mode='evaluate'):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        device = self.device
        user_soc_g = graphs['user_soc_g'].to(torch.device(device))
        user_sim_g = graphs['user_sim_g'].to(torch.device(device))
        item_sim_g = graphs['item_sim_g'].to(torch.device(device))

        # evaluate pos sample
        t0 = time.time()
        pos_users, pos_scores = [], []
        for i, input_list in enumerate(tqdm.tqdm(eval_pos_loader, desc="eval pos_s", smoothing=0, mininterval=1.0)):
            ## sample step
            batch_users, batch_items = list(input_list[0].numpy()), list(input_list[1].numpy())
            self.model.item_users, self.model.item_users_lens = self.history_sample(batch_items, self.item_neigh, batch_users, mode='eval', Type='item')
            self.model.user_items, self.model.user_items_lens = self.history_sample(batch_users, self.user_neigh, batch_items, mode='eval', Type='user')
            self.model.friends_items, self.model.friends_lens = self.friend_sample(batch_users, self.user_friends, self.user_neigh)

            ## eval
            input_list = [x.to(device) for x in input_list]
            pred = self.model(*input_list, user_soc_g, user_sim_g, item_sim_g, mode='eval')
            pos_users.extend(batch_users)
            pos_scores.extend(list(pred.data.cpu().numpy()))
        t1 = time.time()
        
        # evaluate neg sample
        t2 = time.time()
        neg_users, neg_scores = [], []
        for i, input_list in enumerate(tqdm.tqdm(eval_neg_loader, desc="eval neg_s", smoothing=0, mininterval=1.0)):
            ## sample step
            batch_users, batch_items = list(input_list[0].numpy()), list(input_list[1].numpy())
            self.model.item_users, self.model.item_users_lens = self.history_sample(batch_items, self.item_neigh, batch_users, mode='eval', Type='item')
            self.model.user_items, self.model.user_items_lens = self.history_sample(batch_users, self.user_neigh, batch_items, mode='eval', Type='user')
            self.model.friends_items, self.model.friends_lens = self.friend_sample(batch_users, self.user_friends, self.user_neigh)

            ## eval
            input_list = [x.to(device) for x in input_list]
            pred = self.model(*input_list, user_soc_g, user_sim_g, item_sim_g, mode='eval')
            neg_users.extend(batch_users)
            neg_scores.extend(list(pred.data.cpu().numpy()))
        t3 = time.time()
        
        evaluate_result = "\n"
        pos_df = pd.DataFrame({'uid':pos_users, 'score':pos_scores})
        pos_df.sort_values(by=['uid'], ascending=False, inplace=True)
        pos_res = pos_df.groupby('uid')['score'].apply(list).to_dict()
        neg_df = pd.DataFrame({'uid':neg_users, 'score':neg_scores})
        neg_res = neg_df.groupby('uid')['score'].apply(list).to_dict()

        res_hr, res_ndcg, evaluate_result = self.get_metric(pos_res, neg_res, ks=self.eval_ks)

        select_k = self.eval_ks[2]
        print(mode, "pos cost: ", t1-t0, " neg cost: ", t3-t2, "; result, ", "HR@"+str(select_k)+": ", res_hr[2], " NDCG@"+str(select_k)+": ", res_ndcg[2])
        return evaluate_result, [res_hr[2], res_ndcg[2]]  

    