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


def conv1d_transpose():
    obj = Jelly(paddle_api=paddle.nn.functional.conv1d_transpose, torch_api=torch.nn.functional.conv_transpose1d)
    data = randtool("float", 0, 1, [1, 1, 1])
    paddle_param = get_dict(x=data, weight=np.ones(shape=[1, 1, 1]).astype("float64") * 0.3, bias=None, stride=1,
                            padding=0, output_padding=0, groups=1, dilation=1, output_size=None, data_format='NCL', name=None)
    torch_param = get_dict(input=data, weight=np.ones(shape=[1, 1, 1]).astype("float64") * 0.3, bias=None, stride=1, padding=0,
                           output_padding=0, groups=1, dilation=1)
    obj.set_paddle_param(paddle_param)
    obj.set_torch_param(torch_param)
    obj.run()


if __name__ == '__main__':
    conv1d_transpose()
