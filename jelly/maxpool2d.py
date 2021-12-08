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


def maxpool2d():
    obj = Jelly(paddle_api=paddle.nn.MaxPool2D, torch_api=torch.nn.MaxPool2d)
    data = randtool("float", 0, 1, [1, 1, 1, 2])
    paddle_param = get_dict(data=data, kernel_size=1, stride=None, padding=0, return_mask=False, ceil_mode=False, name=None)
    torch_param = get_dict(data=data, kernel_size=1, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.run()


if __name__ == '__main__':
    maxpool2d()