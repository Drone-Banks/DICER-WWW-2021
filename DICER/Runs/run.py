proj_path = "/....../DICER/" # set the absolute path of this project

import sys, argparse
sys.path.append(proj_path)

from Runs.rank_task import Run
from configparser import ConfigParser
cfg = ConfigParser()


import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Welcome to the Experiment Platform Entry')
    parser.add_argument('--data_name', nargs='?', help='data name')
    parser.add_argument('--model_name', nargs='?', help='model name')
    parser.add_argument('--timestamp', default=None, nargs='?', help='timestamp')

    args = parser.parse_args()
    data = args.data_name
    model = args.model_name
    timestamp = args.timestamp

    if timestamp == None:
        mode = 'train'
    else:
        mode = 'test'

    # ======= get the running setting ========
    cfg.read(proj_path+'Runs/configurations/'+ data +'/'+ data+'_'+model+'.ini') 

    print(model)
    # ======= run the main file ============
    
    Run(DataSettings   = dict(cfg.items("DataSettings")),
        ModelSettings  = dict(cfg.items("ModelSettings")),
        TrainSettings  = dict(cfg.items("TrainSettings")),
        ResultSettings = dict(cfg.items("ResultSettings")),
        mode=mode,
        timestamp=timestamp
        )

