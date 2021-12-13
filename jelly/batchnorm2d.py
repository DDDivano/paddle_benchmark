#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import paddle
import torch
import numpy as np
from jelly import Jelly
from jelly import randtool
from jelly import get_dict


def batchnorm2d():
    obj = Jelly(paddle_api=paddle.nn.BatchNorm2D, torch_api=torch.nn.BatchNorm2d)
    data = randtool("float", 0, 1, [2, 1, 1, 1])
    paddle_param = get_dict(data=data, num_features=1, momentum=0.9, epsilon=1e-05, weight_attr=None,
                            bias_attr=None, data_format='NCHW', name=None)
    torch_param = get_dict(data=data, num_features=1, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True, device=None, dtype=None)
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.run()


if __name__ == '__main__':
    batchnorm2d()
