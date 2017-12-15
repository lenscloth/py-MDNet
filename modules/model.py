import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import grid_sample


def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.iteritems():
            if p is None:
                continue
            
            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k
            
            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))

class AffineSpatialTransform(nn.Module):
    def __init__(self, OH, OW):
        super(AffineSpatialTransform, self).__init__()
        hs = torch.linspace(-1, 1, steps=OH)
        ws = torch.linspace(-1, 1, steps=OW)

        hs = hs.unsqueeze(1).expand(OH, OW)
        ws = ws.unsqueeze(0).expand(OH, OW)
        ones = torch.ones(OH, OW)

        t = torch.stack([ws, hs, ones], dim=0)
        self.register_buffer("t", t)
        self.h = OH
        self.w = OW

        # vertices = torch.FloatTensor([[-1, -1, 1],
        #                            [-1, 1, 1],
        #                            [1, 1, 1],
        #                            [1, -1, 1]]).unsqueeze(2)
        # self.register_buffer("vertices", vertices)

    def forward(self, input, theta):
        theta = theta.unsqueeze(1)
        t = Variable(self.t.view(1, 3, -1))
        #v = Variable(self.vertices)

        source_grid = torch.matmul(theta, t)
        source_grid = source_grid.squeeze(1).view(input.size(0), 2, self.h, self.w)
        source_grid = source_grid.permute(0, 2, 3, 1)
        out = grid_sample(input, source_grid)

        # theta = theta.squeeze(1)
        # vertices_out = torch.matmul(theta, v).squeeze(3)

        # ih = input.size(2)
        # iw = input.size(3)
        # x_coord = (vertices_out[..., 0] + 1) * iw / 2
        # y_coord = (vertices_out[..., 1] + 1) * ih / 2
        # vertices_out = torch.stack((x_coord, y_coord), dim=2)

        return out


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1, en_stn=False, zero_init=True):
        super(MDNet, self).__init__()
        self.K = K
        self.en_stn = en_stn
        self.stn = AffineSpatialTransform(5, 5)
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv_loc1', nn.Sequential(nn.Conv2d(256, 64, kernel_size=3),
                                            nn.ReLU())),
                ('conv_loc2', nn.Sequential(nn.Conv2d(64, 6, kernel_size=3))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU())),
                ('fc4',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512 * 3 * 3, 512),
                                        nn.ReLU())),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU()))]))
        if zero_init:
            self.layers[3][0].weight.data = torch.zeros(6, 64, 3, 3)
            self.layers[3][0].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5), 
                                                     nn.Linear(512, 2)) for _ in range(K)])
        
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d'%(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.iteritems():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False
 
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.iteritems():
            if p.requires_grad:
                params[k] = p
        return params
    
    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        #
        # forward model from in_layer to out_layer

        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                if name == "conv_loc1":
                    if self.en_stn:
                        p = module(x)
                elif name == "conv_loc2":
                    if self.en_stn:
                        p = module(p)
                        p = p.view(x.size(0), 2, 3)
                elif name == "conv3":
                    if self.en_stn:
                        x = self.stn(x, p)
                    x = module(x)
                    x = x.view(x.size(0), -1)
                else:
                    x = module(x)

                if name == out_layer:
                    return x
        
        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x)
    
    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)
    
    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]
        
        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i*4]['weights'].item()[0]

            if i == 2:
                i = i + 2
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:,0])


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
 
    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score)[:,1]
        neg_loss = -F.log_softmax(neg_score)[:,0]
        
        loss = pos_loss.sum() + neg_loss.sum()
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        
        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()
        
        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):
        
        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)
        
        return prec.data[0]
