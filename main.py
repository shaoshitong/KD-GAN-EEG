import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time
import torch.optim as optim
import torch
from dataprocessing.dataset import get_data_deap,get_data_seed
from config import get_config
from models.STGAN import STGAN

if __name__ == '__main__':
    torch.cuda.empty_cache()
    total_log = []
    config=get_config()
    if config.DATA.DATASET == "deap":
        total = 32
    elif config.DATA.DATASET == "seed":
        total = 15
    else:
        raise NotImplementedError("not improve this dataset")
    for i in range(total):
        print('*' * 20+f'target subject: {i}'+'*'*20)
        index = np.arange(0, total, 1).tolist()
        del index[i]
        if config.DATA.DATASET == "deap":
            sample_path = config.DATA.DEAP_DATA_PATH
            source_loader, target_loader,source_sample = get_data_deap(sample_path=sample_path,
                                                         index=index,
                                                         i=i,
                                                         num_classes=config.MODEL.DEAP.NUM_CLASSES,
                                                         batchsize=config.MODEL.DEAP.BATCH_SIZE)
            Model = STGAN(config.MODEL.DEAP.IN_FEATURES,config.MODEL.DEAP.HIDDEN_DIM,config.MODEL.DEAP.OUT_FEATURES,config.MODEL.DEAP.CRITIC,
                          epoch=config.EPOCH,warmup_epoch=config.WARMUP_EPOCH)
        elif config.DATA.DATASET == "seed":
            sample_path = config.DATA.SEED_DATA_PATH
            source_loader, target_loader,source_sample = get_data_seed(sample_path=sample_path,
                                                         index=index,
                                                         i=i,
                                                         num_classes=config.MODEL.SEED.NUM_CLASSES,
                                                         batchsize=config.MODEL.SEED.BATCH_SIZE)
            Model = STGAN(config.MODEL.SEED.IN_FEATURES, config.MODEL.SEED.HIDDEN_DIM, config.MODEL.SEED.OUT_FEATURES,config.MODEL.SEED.CRITIC,
                          epoch=config.EPOCH, warmup_epoch=config.WARMUP_EPOCH)

        else:
            raise NotImplementedError("not improve this dataset")
        Model.fit(source_loader,target_loader)
        num=source_sample.shape[0]//(total)
        for i in range(total):
            batch_source_sample=torch.from_numpy(source_sample[i*num:(i+1)*num,...].reshape(num,-1)).float()
            mark=Model.predict(batch_source_sample)
            print(i,round(mark.item(),3),end=" ")


