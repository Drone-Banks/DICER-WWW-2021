import pandas as pd, numpy as np 
import pickle
import tqdm
import math

data_dir = '/home/data_ti5_c/fubr/SocialRecommendation/SocialDatasets/'
data_name = 'Example'
social_data_dir = data_dir + data_name + '/Social/'


### reading data 
data_path = social_data_dir + data_name + '_train.pickle'
train_df = pd.read_pickle(data_path)
data_path = social_data_dir + data_name + '_val.pickle'
val_df = pd.read_pickle(data_path)
data_path = social_data_dir + data_name + '_test.pickle'
test_df = pd.read_pickle(data_path)
data_path = social_data_dir + data_name + '_info.pickle'
info_df = pd.read_pickle(data_path)
data_path = social_data_dir + data_name + '_uu_soc.pickle'
social_df = pd.read_pickle(data_path)


### create graph u_i_u
def create_user_sim_graph(similarity_limit=0.05, Type='cos'):
    user_items_dict = train_df.groupby('user')['item'].apply(list).to_dict()
    total_users = list(user_items_dict.keys())

    user1, user2 = [], []
    for i in tqdm.tqdm(range(len(total_users))):
        u = total_users[i]
        u_items = user_items_dict[u] 
        for j in range(i+1, len(total_users)):
            v = total_users[j]
            v_items = user_items_dict[v]

            hits = 0
            for x in v_items:
                if x in u_items:
                    hits += 1
            
            if Type == 'origin':
                base = min(len(u_items), len(v_items))      # origin
            elif Type == 'cos':
                base = math.sqrt(len(u_items) * len(v_items)) # cos
            elif Type == 'jaccard':
                base = len(set(u_items) | set(v_items))     # jaccard
            else:
                raise ValueError('wrong type' + Type)

            if hits/base >= similarity_limit:
                user1.append(u)
                user2.append(v)
                user1.append(v)
                user2.append(u)

    similarity_df = pd.DataFrame({'user1':user1, 'user2':user2})
    similarity_df.to_pickle(social_data_dir + data_name + '_uu_sim_c'+str(similarity_limit)+'.pickle')
    return similarity_df

### create graph i_u_i
def create_item_sim_graph(similarity_limit=0.05, Type='cos'):
    item_users_dict = train_df.groupby('item')['user'].apply(list).to_dict()
    total_items = list(item_users_dict.keys())

    item1, item2 = [], []
    for i in tqdm.tqdm(range(len(total_items))):
        u = total_items[i]
        u_users = item_users_dict[u] 
        for j in range(i+1, len(total_items)):
            v = total_items[j]
            v_users = item_users_dict[v]

            hits = 0
            for x in v_users:
                if x in u_users:
                    hits += 1

            if Type == 'origin':
                base = min(len(u_users), len(v_users))      # origin
            elif Type == 'cos':
                base = math.sqrt(len(u_users) * len(v_users)) # cos
            elif Type == 'jaccard':
                base = len(set(u_users) | set(v_users))     # jaccard
            else:
                raise ValueError('wrong type' + Type)

            if hits/base >= similarity_limit:
                item1.append(u)
                item2.append(v)
                item1.append(v)
                item2.append(u)

    similarity_df = pd.DataFrame({'item1':item1, 'item2':item2})
    similarity_df.to_pickle(social_data_dir + data_name + '_ii_sim_c'+str(similarity_limit)+'.pickle')

    return similarity_df

similarity_limit = 0.1
# similarity_df = create_user_sim_graph(similarity_limit=similarity_limit)
similarity_df = create_item_sim_graph(similarity_limit=similarity_limit)
