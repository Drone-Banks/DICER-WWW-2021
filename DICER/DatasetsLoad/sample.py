import torch
import tqdm
import time
import random
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy

import dgl


class SampleGenerator(object):
    """ Construct dataset """

    def __init__(self, DataSettings):
        # data settings
        self.DataSettings = DataSettings
        
        # data path
        start_time = time.time()
        self.data_dir = DataSettings['data_dir']
        self.data_name = DataSettings['data_name']
        data_type = DataSettings['data_type']
        self.data_path = self.data_dir + self.data_name + '/' + data_type + '/'
        
        # data settings
        self.train_neg_num = eval(DataSettings['train_neg_num'])
        self.eval_neg_num = eval(DataSettings['eval_neg_num'])
        self.graph_types = eval(DataSettings['graph_types'])

        # read data
        print("=========== reading", self.data_name, "data ===========")
        self._get_main_data()        
        self._get_negative_sample()

        print(f'read pickle file cost {time.time()-start_time} seconds') 
    
   
    def _get_main_data(self):
        self.info_df = pd.read_pickle(self.data_path+self.data_name+'_info.pickle')
        self.train_df = pd.read_pickle(self.data_path+self.data_name+'_train.pickle')
        self.val_df   = pd.read_pickle(self.data_path+self.data_name+'_val.pickle')
        self.test_df  = pd.read_pickle(self.data_path+self.data_name+'_test.pickle')

        if 'user_soc_g' in self.graph_types or 'user_g' in self.graph_types:
            self.uu_soc_df = pd.read_pickle(self.data_path+self.data_name+'_uu_soc.pickle')
            self.user_friend_dict = self.uu_soc_df.groupby('user')['friend'].apply(list).to_dict()
        if 'user_sim_g' in self.graph_types or 'user_g' in self.graph_types:
            uu_sim_limit = self.DataSettings['uu_sim_limit']
            self.uu_sim_df = pd.read_pickle(self.data_path+self.data_name+'_uu_sim_'+uu_sim_limit+'.pickle')
        if 'item_sim_g' in self.graph_types:
            ii_sim_limit = self.DataSettings['ii_sim_limit']
            self.ii_sim_df = pd.read_pickle(self.data_path+self.data_name+'_ii_sim_'+ii_sim_limit+'.pickle')

        self.user_neigh_dict = self.train_df.groupby('user')['item'].apply(list).to_dict()
        self.item_neigh_dict = self.train_df.groupby('item')['user'].apply(list).to_dict()
        self.user_num, self.candidate_num = self.info_df['user_num'][0], self.info_df['candidate_num'][0]
        self.train_size, self.val_size, self.test_size= self.train_df.shape[0], self.val_df.shape[0], self.test_df.shape[0]

    def _get_negative_sample(self):
        data_file = open(self.data_dir+self.data_name+'/'+self.data_name+"_negative_"+str(self.eval_neg_num)+".pickle", 'rb')
        neg_dict = pickle.load(data_file)

        neg_users, neg_items = [], []
        for u in neg_dict.keys():
            tmp_items = neg_dict[u]
            neg_users.extend([u]*len(tmp_items))
            neg_items.extend(tmp_items)
        self.eval_neg = pd.DataFrame({'user':neg_users, 'item':neg_items})

    def generateTrainNegative(self, combine=True):
        bias_id = 1
        num_negatives = self.train_neg_num
        num_items = self.candidate_num
        neg_users, neg_items = [], []
        for row in self.train_df.iterrows():
            u, i = row[1]['user'], row[1]['item']
            for _ in range(num_negatives):
                j = np.random.randint(bias_id, num_items+bias_id)
                while j in self.user_neigh_dict[u]:
                    j = np.random.randint(bias_id, num_items+bias_id)
                neg_users.append(u)
                neg_items.append(j)
        train_neg = pd.DataFrame({'user':neg_users, 'item':neg_items})
        train_neg['rating'] = 0
        train_pos = deepcopy(self.train_df)
        train_pos['rating'] = 1
        self.train_data = pd.concat([train_pos, train_neg], ignore_index=True)

    def _sample_graphs(self, u, v, sample_num):
        node_dict = pd.DataFrame({'u':u, 'v':v}).groupby('u')['v'].apply(list).to_dict()
        new_u, new_v = [], []
        for u in node_dict.keys():
            tmp_vs = node_dict[u]
            tmp_vs = random.sample( tmp_vs, min(len(tmp_vs), sample_num) )
            new_u.extend([u]*len(tmp_vs))
            new_v.extend(tmp_vs)
        return new_u, new_v

    def generateGraphs(self, sample=False):
        graphs = {}
        for graph_type in self.graph_types:
            if graph_type == 'user_soc_g':
                u, v = self.uu_soc_df['user'].tolist(), self.uu_soc_df['friend'].tolist()
                max_nodes = self.user_num+1
                if sample:
                    user_soc_sample_num = eval(self.DataSettings['user_soc_sample_num'])
                    u, v = self._sample_graphs(u, v, user_soc_sample_num)
                user_soc_g = self._create_graph( u, v, max_nodes=max_nodes )
                user_soc_g.ndata['id'] = torch.LongTensor(np.arange(max_nodes))
                graphs['user_soc_g'] = user_soc_g
            elif graph_type == 'user_sim_g':
                u, v = self.uu_sim_df['user1'].tolist(), self.uu_sim_df['user2'].tolist()
                max_nodes = self.user_num+1
                if sample:
                    user_sim_sample_num = eval(self.DataSettings['user_sim_sample_num'])
                    u, v = self._sample_graphs(u, v, self.user_sim_sample_num)
                user_sim_g = self._create_graph( u, v, max_nodes=max_nodes)
                user_sim_g.ndata['id'] = torch.LongTensor(np.arange(max_nodes))
                graphs['user_sim_g'] = user_sim_g
            elif graph_type == 'item_sim_g':
                u, v = self.ii_sim_df['item1'].tolist(), self.ii_sim_df['item2'].tolist()
                max_nodes = self.candidate_num+1
                if sample:
                    item_sim_sample_num = eval(self.DataSettings['item_sim_sample_num'])
                    u, v = self._sample_graphs(u, v, self.item_sim_sample_num)
                item_sim_g = self._create_graph( u, v, max_nodes=max_nodes)
                item_sim_g.ndata['id'] = torch.LongTensor(np.arange(max_nodes))
                graphs['item_sim_g'] = item_sim_g
            else:
                raise ValueError('unknow graph type name: ' + graph_type)
        return graphs

    def _create_graph(self, u, v, max_nodes, self_loop=True):
        g = dgl.graph((u,v), num_nodes=max_nodes)
        if self_loop:
            g = dgl.transform.remove_self_loop(g)
            g.add_edges(g.nodes(), g.nodes())

        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)
        return g
    