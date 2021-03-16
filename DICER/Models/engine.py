import numpy as np
import math
import torch

class BCEFocalLoss(torch.nn.Module):
    
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = 1e-7
 
    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        pt = pt.clamp(self.eps, 1. - self.eps)

        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt).clamp(min=-100) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt).clamp(min=-100)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def use_optimizer(model, TrainSettings):
    optimizer_type = TrainSettings['optimizer']
    learning_rate = eval(TrainSettings['learning_rate'])
    weight_decay  = eval(TrainSettings['weight_decay'])
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'rmse':
        return torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, alpha=0.9)
    else:
        raise ValueError('unknow optimizer_type name: ' + optimizer_type)

def use_criterion(TrainSettings):
    criterion_type = TrainSettings['criterion']

    if criterion_type == 'mse':
        return torch.nn.MSELoss()
    elif criterion_type == 'bce':
        return torch.nn.BCELoss()
    elif criterion_type == 'ce':
        return torch.nn.CrossEntropyLoss()
    elif criterion_type == 'focal':
        return BCEFocalLoss()
    else:
        raise ValueError('unknow criterion_type name: ' + criterion_type)



    
class Engine(object):

    def __init__(self, TrainSettings, ModelSettings, ResultSettings):
        self.modle_settings = ModelSettings  # model configuration
        self.optimizer = use_optimizer(self.model, TrainSettings) 
        self.criterion = use_criterion(TrainSettings) 
        self.device = TrainSettings['device']
    
    def get_metric(self, pos_res, neg_res, ks):
        res_hr, res_ndcg, evaluate_result = [], [], "\n"
        for k in ks:
            hr, ndcg = self.get_rank_metric(pos_res, neg_res, top_k=k) 
            res_hr.append(hr)
            res_ndcg.append(ndcg)
            evaluate_result += "HR@" + str(k) + ": " + str(hr) +"\n"
            evaluate_result += "NDCG@" + str(k) + ": " + str(ndcg) +"\n"
        return res_hr, res_ndcg, evaluate_result

    def getIdcg(self, length):
        idcg = 0.0
        for i in range(length):
            idcg = idcg + math.log(2) / math.log(i + 2)
        return idcg

    def getDcg(self, value):
        dcg = math.log(2) / math.log(value + 2)
        return dcg

    def getHr(self, value):
        hit = 1.0
        return hit

    def get_rank_metric(self, pos_score, neg_score, top_k):
        
        hr_list, ndcg_list = [], []
        for u in pos_score.keys():
            # user_pos = pos_score[u]
            # for p_s in user_pos:
            #     pos_list = [p_s]
            pos_list = pos_score[u]
            neg_list = neg_score[u]
            pos_len = len(pos_list)
            target_length = min(pos_len, top_k)

            total_list = pos_list + neg_list
            sort_index = np.argsort(total_list)
            sort_index = sort_index[::-1]

            user_hr_list = []
            user_ndcg_list = []
            hits_num = 0
            for idx in range(top_k):
                ranking = sort_index[idx]
                if ranking < pos_len:
                    hits_num += 1
                    user_hr_list.append(self.getHr(idx))
                    user_ndcg_list.append(self.getDcg(idx))
            idcg = self.getIdcg(target_length)

            tmp_hr = np.sum(user_hr_list) / target_length
            tmp_ndcg = np.sum(user_ndcg_list) / idcg
            hr_list.append(tmp_hr)
            ndcg_list.append(tmp_ndcg)
        return np.mean(hr_list), np.mean(ndcg_list)



        
        
