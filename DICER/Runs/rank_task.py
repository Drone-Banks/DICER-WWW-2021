proj_path = "/....../DICER/" # set the absolute path of this project

import sys, os
sys.path.append(proj_path)

import numpy as np
import random
import time
import torch
from torch.utils.data import DataLoader

from DatasetsLoad.sample import SampleGenerator
from DatasetsLoad.dataset import PointDataset, RankDataset

from Models.Social.final import FinalEngine


def get_engine(name, Sampler, DataSettings, TrainSettings, ModelSettings, ResultSettings):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    print("=========== model: " + name + " ===========") 
    if name == 'final':
        return FinalEngine(Sampler, DataSettings, TrainSettings, ModelSettings, ResultSettings)
    else:
        raise ValueError('unknow model name: ' + name)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Run(DataSettings, ModelSettings, TrainSettings, ResultSettings,
        mode='train',
        timestamp=None):

    ## =========== setting init ===========
    setup_seed(817) # random seed

    model_name = ModelSettings['model_name']
    save_dir = ResultSettings['save_dir']
    save_dir = save_dir + model_name[0].upper()+model_name[1:] + '/'
    epoch = eval(TrainSettings['epoch'])
    batch_size = eval(TrainSettings['batch_size'])
    s_batch_size = eval(TrainSettings['s_batch_size'])


    ## =========== data  init ===========
    Sampler = SampleGenerator(DataSettings)
    graphs = Sampler.generateGraphs()

    print('User count: %d. Item count: %d. ' % (Sampler.user_num, Sampler.candidate_num))
    print('Without Negatives, Train count: %d. Validation count: %d. Test count: %d' % (Sampler.train_size, Sampler.val_size, Sampler.test_size))


    ## =========== model init ===========
    Engine = get_engine(model_name, Sampler, DataSettings, TrainSettings, ModelSettings, ResultSettings)

    ## =========== train || inference ===
    if timestamp == None:
        timestamp = time.time()
    localtime = str(time.asctime( time.localtime(int(timestamp)) ))
    with open(save_dir + model_name + "_" + str(int(timestamp)) +".txt", "a") as f:
        model_save_dir = save_dir+'files/'+str(int(timestamp))+'/'
        
        f.write('\n\n\n'+'========== ' + localtime + " "+ str(int(timestamp)) + ' =========='+'\n')
        f.write(str(DataSettings)+'\n'+str(ModelSettings)+'\n'+str(TrainSettings)+'\n')
        f.write(str(Engine.model))
        f.write('\n')

        if mode == "train": # train mode
            print("=========== Training Start ===========") 
            val_hr, val_ndcg = 0, 0
            test_hr, test_ndcg = 0, 0
            best_result = ""
            endure_count = 0
            early_stop_step = 10
            pre_train_epoch = 0

            for epoch_i in range(0, epoch):
                
                ### train
                Sampler.generateTrainNegative(combine=True)
                train_loader = DataLoader(PointDataset(Sampler.train_data), batch_size=batch_size, shuffle=True, num_workers=0)
                Engine.train(train_loader, graphs, epoch_i)
                
                ### early stop
                if epoch_i >= pre_train_epoch:
                    ### test
                    test_pos_loader = DataLoader(RankDataset(Sampler.test_df), batch_size=s_batch_size, shuffle=True, num_workers=0)
                    test_neg_loader = DataLoader(RankDataset(Sampler.eval_neg), batch_size=s_batch_size, shuffle=True, num_workers=0)
                    result, res = Engine.evaluate(test_pos_loader, test_neg_loader, graphs, epoch_i)
                    tmp_hr, tmp_ndcg = res

                    if tmp_ndcg > val_ndcg:
                        val_hr, val_ndcg = tmp_hr, tmp_ndcg
                        endure_count = 0
                        # result, res = Engine.evaluate(Sampler, graphs, epoch_i, mode='test')
                        # tmp_hr, tmp_ndcg = res
                        # if tmp_ndcg > test_ndcg:
                        #     test_hr, test_ndcg = tmp_hr, tmp_ndcg
                        best_result = result
                        test_epoch = epoch_i
                        print(str(int(timestamp))+' new test result:',  best_result)
                        # save log
                        f.write('epoch: ' + str(epoch_i) + '\n')
                        f.write(result+'\n')
                        # save model
                        if not os.path.exists(model_save_dir):
                            os.makedirs(model_save_dir)
                        torch.save(Engine.model, f'{model_save_dir}{model_name}.pt')
                    else:
                        endure_count += 1
                    
                    if endure_count > early_stop_step:
                        break
            
            # finish training
            print(str(int(timestamp))+' best test result:', best_result)
            f.write('test results(epoch: '+str(test_epoch)+' timestamp: '+str(int(timestamp))+'):\n'+best_result+'\n')

            print("test the 1000 sample metric")
            Sampler.eval_neg_num = 1000
            Sampler._get_negative_sample()
            Engine.model = torch.load(f'{model_save_dir}{model_name}.pt').to(TrainSettings['device'])
            Engine.eval_ks = [5, 10, 15]
            test_pos_loader = DataLoader(RankDataset(Sampler.test_df), batch_size=s_batch_size, shuffle=True, num_workers=0)
            test_neg_loader = DataLoader(RankDataset(Sampler.eval_neg), batch_size=s_batch_size, shuffle=True, num_workers=0)
            result, res = Engine.evaluate(test_pos_loader, test_neg_loader, graphs, epoch_id=0)

            print('test results( timestamp: '+str(int(timestamp))+'):\n', result)
            f.write('test results( timestamp: '+str(int(timestamp))+'):\n'+result+'\n')

        else:
            print("=========== Inference Start ===========") 

            print("test the 1000 sample metric")
            Sampler.eval_neg_num = 1000
            Sampler._get_negative_sample()

            Engine.model = torch.load(f'{model_save_dir}{model_name}.pt').to(TrainSettings['device'])
            Engine.eval_ks = [5, 10, 15]
            test_pos_loader = DataLoader(RankDataset(Sampler.test_df), batch_size=s_batch_size, shuffle=True, num_workers=0)
            test_neg_loader = DataLoader(RankDataset(Sampler.eval_neg), batch_size=s_batch_size, shuffle=True, num_workers=0)
            result, res = Engine.evaluate(test_pos_loader, test_neg_loader, graphs, epoch_id=0)

            print('test results( timestamp: '+str(int(timestamp))+'):\n', result)
            f.write('test results( timestamp: '+str(int(timestamp))+'):\n'+result+'\n')
