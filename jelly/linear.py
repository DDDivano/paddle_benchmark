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


def linear():
    obj = Jelly(paddle_api=paddle.nn.Linear, torch_api=torch.nn.Linear)
    data = randtool("float", 0, 1, [1, 1])
    paddle_param = get_dict(data=data, in_features=1, out_features=1, weight_attr=None, bias_attr=None, name=None)
    torch_param = get_dict(data=data, in_features=1, out_features=1, bias=True, device=None, dtype=None)
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.run()


if __name__ == '__main__':
    linear()