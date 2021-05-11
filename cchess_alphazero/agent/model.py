import hashlib
import json
import os
from logging import getLogger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, ActionLabelsBlack

logger = getLogger(__name__)

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



class CChessModel(torch.nn.Module):

    def __init__(self, config: Config):
        super(CChessModel, self).__init__()
        self.config = config
        self.mc = self.config.model
        self.digest = None
        self.n_labels = len(ActionLabelsRed)
        self.graph = None
        self.api = None

        # common layers
        self.conv1 = torch.nn.Conv2d(self.mc.input_depth,self.mc.cnn_filter_num,kernel_size=self.mc.cnn_first_filter_size)
        self.residual_conv_list = [torch.nn.Conv2d(self.mc.cnn_filter_num, self.mc.cnn_filter_num, kernel_size = self.mc.cnn_filter_size, padding = 1) for _ in range(self.mc.res_layer_num * 2)]
        # self.common_batchnorm = torch.nn.BatchNorm2d(num_features,momentum=0.99, affine=True, track_running_stats=True)
        
        # action policy layers
        self.policy_conv1 = torch.nn.Conv2d(self.mc.cnn_filter_num, 4, kernel_size= 1)
        self.policy_flatten = torch.nn.Flatten()
        self.act_fc1 = nn.Linear(120, self.n_labels)


        # state value layers
        self.value_conv1 = torch.nn.Conv2d(self.mc.cnn_filter_num,2,kernel_size= 1)
        self.value_fc1 = nn.Linear(60, self.mc.value_fc_size)
        self.value_fc2 = nn.Linear(self.mc.value_fc_size, 1)

        # self.model = self
        
    def _build_residual_block(self, x, i):
        mc = self.config.model
        in_x = x
        res_name = "res" + str(i)
        x = self.residual_conv_list[i*2](x)
        x = F.relu(x)
        x = self.residual_conv_list[i*2+1](x)
        x = torch.add(in_x, x)
        x = F.relu(x)
        return x

    def forward(self, x):
        
        x = torch.from_numpy(x)

        # common layers
        x = F.relu(self.conv1(x))
        for i in range(self.mc.res_layer_num):
            x = self._build_residual_block(x, i)
        res_out = x

        # action policy layers
        x_act = F.relu(self.policy_conv1(x))
        x_act = x_act.view(-1, 120)
        x_act = F.log_softmax(self.act_fc1(x_act))

        # state value layers
        x_val = F.relu(self.value_conv1(x))
        x_val = x_val.view(-1, 60)
        x_val = F.relu(self.value_fc1(x_val))
        x_val = F.tanh(self.value_fc2(x_val))

        return x_act, x_val


    def load(self, model_path):
        if os.path.exists(model_path) :
            logger.debug(f"loading model from {model_path}")
            self = torch.load(model_path)

            return True
        else:
            logger.debug(f"model files does not exist at {model_path}")
            return False

    def save(self, model_path):
        logger.debug(f"save model to {model_path}")
        torch.save(self, model_path)
        

    def get_pipes(self, num=1, api=None, need_reload=True):
        if self.api is None:
            self.api = CChessModelAPI(self.config, self)
            self.api.start(need_reload)
        return self.api.get_pipe(need_reload)

    def close_pipes(self):
        if self.api is not None:
            self.api.close()
            self.api = None

    def predict_on_batch(self, input):
        # gpu
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.model.to(device)
        with torch.no_grad():
            # input=input.to(device)
            out = self(input)
            return out[0].data, out[1].data

