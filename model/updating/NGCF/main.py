import world
import utils
from utils import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import pickle
from parse import parse_args
import dataloader
import os
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
import register

args = parse_args()

dataset = dataloader.Loader(path="../data/"+world.dataset)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)
va_metric_max = 0
patience_count = 0
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.dataset + "-" + str(args.train_set) + "-" + world.model_name)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")
start = time.time()
try:
    for epoch in range(world.TRAIN_epochs):
        if epoch %10 == 0:
            cprint("[TEST]")
            for i in range(3):
                Procedure.Test(dataset, Recmodel, epoch, i, w, world.config['multicore'])
            
            cprint("[VAL]")
            i = 3
            va_metric = Procedure.Test(dataset, Recmodel, epoch, i, w, world.config['multicore'])
            va_metric_current = va_metric['ndcg'][0]
            if (va_metric_current > va_metric_max) and (epoch > 0):
                va_metric_max = va_metric_current
                torch.save(Recmodel.state_dict(), weight_file)
                patience_count = 0
            else:
                patience_count += 1
            print(f'patience_count is {patience_count}')
            if patience_count == 10:
                break

        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')


finally:
    if world.tensorboard:
        w.close()
end = time.time()
train_time = end - start
print("train:{:.0f}s".format(train_time))
cprint("[TEST]")
Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))

result_file = '../result/'
if not os.path.exists(result_file):
    os.makedirs(result_file)
with open(result_file + '{}-{}.txt'.format(str(args.model), str(args.dataset)), 'a') as f:
    f.write(str(vars(args)))
    f.write('\n')
    for i in range(3):
        # 0: cold  1: warm  2: all
        result = Procedure.Test(dataset, Recmodel, epoch, i, w, world.config['multicore'])
        f.write('%.4f %.4f  | ' % (result['recall'][1], result['ndcg'][1]))
        if i == 2:
            f.write('\n')
